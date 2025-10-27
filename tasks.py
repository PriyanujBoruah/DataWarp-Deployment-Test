# tasks.py

import os
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype, is_bool_dtype
import time
from memory_profiler import memory_usage
import json
import re
import numpy as np
import sys
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from category_encoders.hashing import HashingEncoder
from rq import get_current_job
from annoy import AnnoyIndex
import duckdb
from datetime import datetime
import colormap
import uuid
import logging
from scipy.stats import chi2_contingency, norm
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now this import will succeed because the worker knows where to find the 'app' folder.
from app.utils import _profiled_pipeline_execution, _run_single_benchmark, save_new_state, get_current_state_df, format_bytes, get_column_type_summary

# We also need some specific imports for the execute_pipeline_code_task
from sklearn.preprocessing import PowerTransformer


logger = logging.getLogger(__name__)

import math

def sanitize_for_json(obj):
    """
    Recursively walks a dictionary or list and converts any NaN or Inf
    values to None, which is JSON-serializable as `null`.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

# ===================================================================
# === TASK DEFINITIONS
# ===================================================================

# --- TASK 1: For `run_pipelines_for_comparison` ---
def run_pipelines_for_comparison_task(history_dir_path, pipeline_a_spec, pipeline_b_spec, use_raw_dataset):
    """
    The background task that robustly runs two pipelines on either the raw
    or current dataset, profiles them for time and memory, and reports progress.
    """
    job = get_current_job()
    job.meta.update({'progress': 0, 'status_message': 'Initializing comparison...'})
    job.save_meta()

    try:
        logger.info(f"WORKER: Starting pipeline comparison. Use Raw Dataset: {use_raw_dataset}")

        # --- LOGIC TO SELECT THE CORRECT INPUT DATAFRAME ---
        if use_raw_dataset:
            initial_df_path = os.path.join(history_dir_path, 'raw.parquet')
            if not os.path.exists(initial_df_path):
                raise FileNotFoundError("Initial 'raw.parquet' dataset not found. Please re-upload the file.")
            initial_df = pd.read_parquet(initial_df_path)
            source_name = "Raw Dataset"
        else:
            # Load the current, interactively cleaned dataset state
            initial_df = get_current_state_df(history_dir_path)
            source_name = "Current Cleaned Dataset"

        if initial_df.empty:
            raise ValueError(f"The selected source ({source_name}) is empty. Cannot run pipelines.")
        # --- END OF DATAFRAME SELECTION LOGIC ---
        
        # --- Profile Pipeline A ---
        pipeline_a_name = pipeline_a_spec.get('name', 'Pipeline A')
        job.meta.update({'progress': 10, 'status_message': f"Running {pipeline_a_name} on {source_name}..."})
        job.save_meta()
        
        logger.info(f"WORKER: Profiling {pipeline_a_name}...")
        start_a = time.time()
        # Use max_usage=True for more efficient memory profiling
        mem_usage_a = memory_usage(
            (_profiled_pipeline_execution, (pipeline_a_spec['steps'], initial_df)),
            interval=0.1, retval=True, max_usage=True
        )
        time_a = time.time() - start_a
        df_a, peak_mem_a = mem_usage_a[1], mem_usage_a[0]
        logger.info(f"WORKER: {pipeline_a_name} finished in {time_a:.2f}s.")

        # --- Profile Pipeline B ---
        pipeline_b_name = pipeline_b_spec.get('name', 'Pipeline B')
        job.meta.update({'progress': 50, 'status_message': f"Running {pipeline_b_name} on {source_name}..."})
        job.save_meta()

        logger.info(f"WORKER: Profiling {pipeline_b_name}...")
        start_b = time.time()
        mem_usage_b = memory_usage(
            (_profiled_pipeline_execution, (pipeline_b_spec['steps'], initial_df)),
            interval=0.1, retval=True, max_usage=True
        )
        time_b = time.time() - start_b
        df_b, peak_mem_b = mem_usage_b[1], mem_usage_b[0]
        logger.info(f"WORKER: {pipeline_b_name} finished in {time_b:.2f}s.")

        # --- Finalizing and Calculating Schema ---
        job.meta.update({'progress': 95, 'status_message': 'Finalizing results and finding common columns...'})
        job.save_meta()

        # Save the processed dataframes for the next stage (benchmarking)
        df_a.to_parquet(os.path.join(history_dir_path, 'comparator_a.parquet'))
        df_b.to_parquet(os.path.join(history_dir_path, 'comparator_b.parquet'))

        common_columns = sorted(list(set(df_a.columns) & set(df_b.columns)))
        
        common_schema = []
        for col_name in common_columns:
            # Use df_a for schema detection as a consistent source
            is_numeric = pd.api.types.is_numeric_dtype(df_a[col_name])
            col_info = {'name': col_name, 'type': 'numeric' if is_numeric else 'categorical'}
            if not is_numeric:
                col_info['cardinality'] = df_a[col_name].nunique()
            common_schema.append(col_info)

        # Return a JSON-serializable dictionary with all the results
        return {
            "result_a": {
                "execution_time": round(time_a, 2),
                "final_shape": {"rows": df_a.shape[0], "columns": df_a.shape[1]},
                "peak_memory_mb": round(peak_mem_a, 2),
            },
            "result_b": {
                "execution_time": round(time_b, 2),
                "final_shape": {"rows": df_b.shape[0], "columns": df_b.shape[1]},
                "peak_memory_mb": round(peak_mem_b, 2),
            },
            "common_schema": common_schema
        }
    except Exception as e:
        logger.error(f"Pipeline comparison task failed: {e}", exc_info=True)
        # Re-raise the exception so RQ marks the job as failed and reports the error
        raise

# --- TASK 2: For `run_benchmark_for_comparison` ---
def run_benchmark_for_comparison_task(history_dir_path, target, problem_type, pipeline_a_spec, pipeline_b_spec, time_a, time_b):
    """
    The background task that robustly runs the PyCaret ML benchmarks on the two 
    pipeline outputs and reports detailed progress.
    """
    job = get_current_job()
    job.meta.update({'progress': 0, 'status_message': 'Initializing ML Benchmark...'})
    job.save_meta()

    try:
        logger.info(f"WORKER: Starting ML benchmark task for target: {target}")
        df_a = pd.read_parquet(os.path.join(history_dir_path, 'comparator_a.parquet'))
        df_b = pd.read_parquet(os.path.join(history_dir_path, 'comparator_b.parquet'))

        # --- Pre-flight Validation Checks ---
        if target not in df_a.columns:
            raise ValueError(f"Pipeline A is invalid. It removed or renamed the benchmark target column: '{target}'.")
        if target not in df_b.columns:
            raise ValueError(f"Pipeline B is invalid. It removed or renamed the benchmark target column: '{target}'.")

        if df_a.empty or df_b.empty:
            raise ValueError("One or both pipelines produced an empty dataset. Cannot run benchmark.")

        # Sanitize column names for PyCaret compatibility
        def sanitize_column_name(col):
            return re.sub(r'[^A-Za-z0-9_]+', '', str(col))

        df_a.columns = [sanitize_column_name(col) for col in df_a.columns]
        df_b.columns = [sanitize_column_name(col) for col in df_b.columns]
        target = sanitize_column_name(target)

        # --- Run Benchmark for Pipeline A ---
        pipeline_a_name = pipeline_a_spec.get('name', 'Pipeline A')
        job.meta.update({'progress': 10, 'status_message': f"Benchmarking {pipeline_a_name}... This may take several minutes."})
        job.save_meta()
        
        logger.info(f"WORKER: Running benchmark for {pipeline_a_name}...")
        df_bench_a = _run_single_benchmark(df_a, target, problem_type, pipeline_a_spec)
        
        # --- Run Benchmark for Pipeline B ---
        pipeline_b_name = pipeline_b_spec.get('name', 'Pipeline B')
        job.meta.update({'progress': 50, 'status_message': f"Benchmarking {pipeline_b_name}... This may also take a while."})
        job.save_meta()

        logger.info(f"WORKER: Running benchmark for {pipeline_b_name}...")
        df_bench_b = _run_single_benchmark(df_b, target, problem_type, pipeline_b_spec)

        # --- Finalizing and Merging Results ---
        job.meta.update({'progress': 95, 'status_message': 'Merging results and generating summary...'})
        job.save_meta()
        
        if problem_type == 'regression':
            metrics_to_keep = ['Model', 'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'RMSLE', 'TT (Sec)']
        else: # Classification
            metrics_to_keep = ['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']

        # Ensure all expected columns exist, filling with NaN if they don't
        df_bench_a = df_bench_a.reindex(columns=metrics_to_keep)
        df_bench_b = df_bench_b.reindex(columns=metrics_to_keep)

        merged_df = pd.merge(df_bench_a, df_bench_b, on='Model', suffixes=('_a', '_b'), how='outer')
        merged_results_json = merged_df.to_dict(orient='records')
        
        # Simple summary text (can be enhanced later)
        summary_text = f"Benchmark complete. Comparing '{pipeline_a_name}' vs. '{pipeline_b_name}' for predicting '{target}'."

        logger.info("WORKER: ML benchmark task finished.")
        return { 
            "merged_benchmark": merged_results_json,
            "summary_text": summary_text
        }
    except Exception as e:
        logger.error(f"Benchmark task failed: {e}", exc_info=True)
        # Re-raise the exception for RQ
        raise
    
# --- TASK 3: For `execute_pipeline_code` ---
def execute_pipeline_code_task(script_code, initial_state_path, history_dir_path):
    """
    The background task that executes a dynamically generated pipeline script.
    """
    # This dictionary will capture the 'final_df' variable created by the script.
    execution_namespace = {}

    # The code to be executed is the full script string, plus a line
    # to call the main function within it and store its return value.
    # repr() ensures the path is correctly formatted as a string literal.
    exec_code = f"{script_code}\nfinal_df = run_pipeline(data_path={repr(initial_state_path)})"
    
    # This dictionary defines all the global modules and functions that the
    # executed script is allowed to access.
    exec_globals = {
        # Modules needed by the pipeline steps:
        'pd': pd,
        'np': np,
        're': re,
        'os': os, # <--- THIS IS THE FIX. The 'os' module is now available.
        
        # Specific classes needed by certain steps:
        'PowerTransformer': PowerTransformer
    }
    
    # Execute the code in a controlled environment.
    # The `exec_globals` are the available tools.
    # The `execution_namespace` is where the results are stored.
    exec(exec_code, exec_globals, execution_namespace)
    
    # Retrieve the resulting DataFrame from the namespace.
    result_df = execution_namespace['final_df']
    
    # Now that the pipeline has run, save the resulting DataFrame as a new state.
    # We explicitly pass the history_dir_path so this works in the background.
    save_new_state(result_df, history_dir_path, description="Executed generated pipeline")
    
    # Return a JSON-serializable dictionary to be stored as the job's result.
    return {
        "message": "Pipeline executed successfully!", 
        "new_row_count": len(result_df), 
        "new_col_count": len(result_df.columns)
    }

def smart_impute_task(history_dir_path):
    """
    The background task that performs a robust, KNN-style imputation on the entire dataset.
    This version uses Pandas exclusively for maximum compatibility and robustness.
    """
    # === Stage 1: Initialization and Progress Reporting ===
    job = get_current_job()
    job.meta.update({'progress': 0, 'status_message': 'Initializing Smart-Impute...'})
    job.save_meta()

    df = get_current_state_df(history_dir_path).copy()
    if df.empty:
        return {"message": "Dataset is empty, nothing to impute.", "rows_affected": 0}

    missing_mask = df.isnull().any(axis=1)
    if not missing_mask.any():
        return {"message": "No missing values found to impute.", "rows_affected": 0}

    # === Stage 2: Data Typing and Pre-calculation of Global Fallbacks ===
    job.meta.update({'progress': 5, 'status_message': 'Analyzing data types and preparing fallbacks...'})
    job.save_meta()
    
    # Using pandas.api.types for robust type checking
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if is_string_dtype(df[col])]
    datetime_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
    boolean_cols = [col for col in df.columns if is_bool_dtype(df[col])]

    global_fallbacks = {}
    for col in numeric_cols: global_fallbacks[col] = df[col].median()
    for col in categorical_cols: 
        mode_series = df[col].mode()
        global_fallbacks[col] = mode_series[0] if not mode_series.empty else None
    for col in datetime_cols: global_fallbacks[col] = df[col].median()
    for col in boolean_cols: 
        mode_series = df[col].mode()
        global_fallbacks[col] = mode_series[0] if not mode_series.empty else None

    # === Stage 3: Preprocessing for Annoy Index (Vectorization) ===
    job.meta.update({'progress': 15, 'status_message': 'Vectorizing data for neighbor search...'})
    job.save_meta()

    df_for_transform = df.copy()
    for col, fallback_value in global_fallbacks.items():
        if fallback_value is not None:
            df_for_transform[col].fillna(fallback_value, inplace=True)

    df_for_transform = pd.DataFrame(df_for_transform.to_dict('list'))
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', HashingEncoder(n_components=16), categorical_cols),
        ('dt', FunctionTransformer(lambda x: x.astype('int64') // 10**9), datetime_cols),
        ('bool', FunctionTransformer(lambda x: x.astype('int8')), boolean_cols)
    ], remainder='drop', n_jobs=1)

    df_transformed = preprocessor.fit_transform(df_for_transform)

    # === Stage 4: Build Nearest Neighbors Index ===
    job.meta.update({'progress': 30, 'status_message': 'Building nearest neighbors index...'})
    job.save_meta()
    
    num_features = df_transformed.shape[1]
    annoy_index = AnnoyIndex(num_features, 'euclidean')
    for i, vector in enumerate(df_transformed):
        annoy_index.add_item(i, vector)
    annoy_index.build(10)

    # === Stage 5: Main Imputation Loop ===
    df_imputed = df.copy()
    rows_to_impute_indices = df.index[missing_mask]
    total_rows_to_process = len(rows_to_impute_indices)
    processed_count = 0
    fallback_count = 0
    smart_imputed_count = 0
    
    for row_index in rows_to_impute_indices:
        row_series = df.loc[row_index]
        neighbors_indices = [idx for idx in annoy_index.get_nns_by_item(row_index, 6) if idx != row_index]
        neighbor_rows = df.iloc[neighbors_indices]

        for col_name in df.columns:
            if pd.notna(row_series[col_name]):
                continue
            
            imputation_value = None
            if col_name in numeric_cols: imputation_value = neighbor_rows[col_name].median()
            elif col_name in categorical_cols: 
                mode_series = neighbor_rows[col_name].mode()
                imputation_value = mode_series[0] if not mode_series.empty else None
            elif col_name in datetime_cols: imputation_value = neighbor_rows[col_name].median()
            elif col_name in boolean_cols: 
                mode_series = neighbor_rows[col_name].mode()
                imputation_value = mode_series[0] if not mode_series.empty else None
            
            if pd.isna(imputation_value):
                fallback_count += 1
                imputation_value = global_fallbacks.get(col_name)
            else:
                # If we didn't need the fallback, it was a successful "smart" imputation
                smart_imputed_count += 1

            if pd.notna(imputation_value):
                df_imputed.at[row_index, col_name] = imputation_value
        
        processed_count += 1
        progress_percent = 30 + (processed_count / total_rows_to_process) * 65
        job.meta.update({'progress': round(progress_percent, 2), 'status_message': f'Imputing row {processed_count} of {total_rows_to_process}...'})
        job.save_meta()

    # === Stage 6: Finalization ===
    job.meta.update({'progress': 98, 'status_message': 'Finalizing and saving new dataset state...'})
    job.save_meta()
    
    save_new_state(df_imputed, history_dir_path, description="Applied Smart-Impute")

    message = (
        f"Successfully imputed missing values in {len(rows_to_impute_indices)} rows. "
        f"Smart imputed {smart_imputed_count} cells"
    )
    if fallback_count > 0:
        message += f" and used a global fallback method for {fallback_count} individual cells."
    else:
        message += "."

    return { "message": message, "rows_affected": len(rows_to_impute_indices) }

def generate_full_report_task(history_dir_path: str):
    """
    The background task that performs a comprehensive EDA for the full data report
    on the CURRENT dataset state, using Pandas, SciPy, and Matplotlib.
    """
    job = get_current_job()
    job.meta.update({'progress': 0, 'status_message': 'Initializing full report...'})
    job.save_meta()

    try:
        # Load the current, cleaned DataFrame from the user's session history.
        df = get_current_state_df(history_dir_path)
    except Exception as e:
        logger.error(f"Failed to load dataset state from {history_dir_path}: {e}", exc_info=True)
        raise ValueError(f"Could not load the current dataset state. Error: {e}")

    if df.empty:
        raise ValueError("The current dataset is empty. Cannot generate a report.")

    report = {}
    total_rows, total_cols = df.shape
    
    # === 1. Global Overview Analysis ===
    job.meta.update({'progress': 5, 'status_message': 'Analyzing global dataset properties...'})
    job.save_meta()

    memory_usage = df.memory_usage(deep=True).sum()
    total_missing = df.isnull().sum().sum()
    total_cells = total_rows * total_cols
    duplicate_rows = df.duplicated().sum()
    
    type_counts = df.dtypes.value_counts().to_dict()
    variable_types = {str(k): int(v) for k, v in type_counts.items()}

    report['header'] = {
        'dataset_name': "Current Cleaned Dataset",
        'generation_time': datetime.now().strftime("%B %d, %Y, %I:%M %p"),
        'summary': f"An in-depth analysis of {total_rows:,} rows and {total_cols} columns."
    }
    report['global_overview'] = {
        'observations': total_rows,
        'variables': total_cols,
        'memory_footprint': format_bytes(memory_usage),
        'total_missing': int(total_missing),
        'missing_percent': f"{(total_missing / total_cells) * 100:.2f}" if total_cells > 0 else "0.00",
        'duplicate_rows': int(duplicate_rows),
        'variable_types': variable_types
    }

    # === 2. Detailed Column-by-Column Analysis ===
    column_details_list = []
    total_columns_to_process = len(df.columns)
    for i, col_name in enumerate(df.columns):
        progress = 10 + (i / total_columns_to_process) * 60
        job.meta.update({'progress': round(progress, 2), 'status_message': f'Analyzing column {i+1}/{total_columns_to_process}: "{col_name}"...'})
        job.save_meta()

        col_series = df[col_name]
        details = {'name': col_name, 'type': str(col_series.dtype)}
        
        common_stats = {
            'missing': int(col_series.isnull().sum()),
            'missing_percent': f"{(col_series.isnull().sum() / total_rows) * 100:.2f}%",
            'unique_values': col_series.nunique(),
            'memory_usage': format_bytes(col_series.memory_usage(deep=True))
        }
        
        if pd.api.types.is_numeric_dtype(col_series):
            details['type_group'] = 'numeric'
            desc = col_series.describe().to_dict()
            numeric_stats = {
                'mean': desc.get('mean'), 'std_dev': desc.get('std'), 'min': desc.get('min'),
                '25%': desc.get('25%'), 'median': desc.get('50%'), '75%': desc.get('75%'),
                'max': desc.get('max'), 'sum': col_series.sum(), 'variance': col_series.var(),
                'skewness': col_series.skew(), 'kurtosis': col_series.kurt(),
                'zeros_count': int((col_series == 0).sum()),
            }
            details['stats'] = {**common_stats, **numeric_stats}
            hist_data, bin_edges = np.histogram(col_series.dropna(), bins=20)
            details['chart_data'] = {"type": "histogram", "data": [{"x": float(bin_edges[i]), "y": int(hist_data[i])} for i in range(len(hist_data))]}

        elif pd.api.types.is_string_dtype(col_series) or pd.api.types.is_categorical_dtype(col_series):
            details['type_group'] = 'categorical'
            desc = col_series.describe().to_dict()
            lengths = col_series.dropna().astype(str).str.len()
            categorical_stats = {
                'top_value': desc.get('top'), 'top_freq': desc.get('freq'),
                'min_length': lengths.min() if not lengths.empty else 0,
                'mean_length': lengths.mean() if not lengths.empty else 0,
                'max_length': lengths.max() if not lengths.empty else 0,
            }
            details['stats'] = {**common_stats, **categorical_stats}
            freq = col_series.value_counts().nlargest(20)
            details['chart_data'] = {"type": "bar", "data": [{"category": str(k), "count": int(v)} for k, v in freq.items()]}

        else: # Datetime, Boolean, or other types
            details['type_group'] = 'other'
            if pd.api.types.is_datetime64_any_dtype(col_series):
                desc = col_series.astype('int64').describe().to_dict()
            else:
                desc = col_series.describe().to_dict()
            details['stats'] = {**common_stats, **desc}
            freq = col_series.value_counts().nlargest(20)
            details['chart_data'] = {"type": "bar", "data": [{"category": str(k), "count": int(v)} for k, v in freq.items()]}

        column_details_list.append(details)
    
    report['column_details'] = column_details_list

    # === 3. Relationships and Interactions Analysis ===
    job.meta.update({'progress': 75, 'status_message': 'Analyzing relationships between variables...'})
    job.save_meta()

    def cramers_v(x, y):
        contingency_table = pd.crosstab(x, y)
        if contingency_table.empty: return 0
        chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        n = contingency_table.sum().sum()
        if n == 0: return 0
        phi2 = chi2 / n
        r, k = contingency_table.shape
        if min(k - 1, r - 1) == 0: return 0
        return np.sqrt(phi2 / min(k - 1, r - 1))

    columns = df.columns
    matrix = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns)
    
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            col1_name, col2_name = columns[i], columns[j]
            col1, col2 = df[col1_name], df[col2_name]
            
            if i == j: continue

            if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                correlation = col1.corr(col2)
                matrix.iloc[i, j] = matrix.iloc[j, i] = correlation if not pd.isna(correlation) else 0
            elif not pd.api.types.is_numeric_dtype(col1) and not pd.api.types.is_numeric_dtype(col2):
                matrix.iloc[i, j] = matrix.iloc[j, i] = cramers_v(col1.dropna(), col2.dropna())
            else:
                numeric_col, cat_col = (col1, col2) if pd.api.types.is_numeric_dtype(col1) else (col2, col1)
                temp_df = pd.concat([numeric_col, cat_col], axis=1).dropna()
                ss_total = np.sum((temp_df[numeric_col.name] - temp_df[numeric_col.name].mean())**2)
                if ss_total > 0:
                    ss_between = np.sum(temp_df.groupby(cat_col.name)[numeric_col.name].count() * (temp_df.groupby(cat_col.name)[numeric_col.name].mean() - temp_df[numeric_col.name].mean())**2)
                    eta_squared = ss_between / ss_total
                    matrix.iloc[i, j] = matrix.iloc[j, i] = np.sqrt(eta_squared)
                else:
                    matrix.iloc[i, j] = matrix.iloc[j, i] = 0
    
    # Format the matrix for the frontend using Matplotlib
    cmap = colormaps.get_cmap('coolwarm')
    normalizer = Normalize(vmin=-1, vmax=1)
    heatmap_data = {"labels": matrix.columns.tolist(), "rows": []}
    
    for index, row in matrix.iterrows():
        formatted_row = []
        for col_name in heatmap_data['labels']:
            val = row[col_name]
            rgba_color = cmap(normalizer(val))
            color_hex = to_hex(rgba_color)
            formatted_row.append({"value": f"{val:.2f}", "color": color_hex})
        heatmap_data["rows"].append(formatted_row)
    report['association_matrix'] = heatmap_data

    job.meta.update({'progress': 100, 'status_message': 'Report generation complete!'})
    job.save_meta()

    return sanitize_for_json(report)

def commit_data_view_task(filepath, sort_conditions, filter_conditions, search_params):
    """
    The background task that applies a view (filters, sorts, search)
    and overwrites the original CSV file.
    """
    con = duckdb.connect()
    
    # Use the original uploaded file as the source
    base_query = f"FROM read_csv_auto('{filepath}')"
    
    # --- Build WHERE Clause (combining filter and search) ---
    where_clauses = []
    params = []

    # 1. Add filter conditions
    for cond in filter_conditions:
        col, op, val = cond.get('column'), cond.get('operator'), cond.get('value')
        if not col or not op: continue
        
        if op in ['=', '!=', '>', '<', '>=', '<=']:
            where_clauses.append(f'"{col}" {op} ?')
            params.append(val)
        elif op == 'contains':
            where_clauses.append(f'"{col}" ILIKE ?')
            params.append(f'%{val}%')
        elif op == 'starts_with':
            where_clauses.append(f'"{col}" ILIKE ?')
            params.append(f'{val}%')
        elif op == 'ends_with':
            where_clauses.append(f'"{col}" ILIKE ?')
            params.append(f'%{val}%')
        elif op == 'is_null':
            where_clauses.append(f'"{col}" IS NULL')
        elif op == 'is_not_null':
            where_clauses.append(f'"{col}" IS NOT NULL')

    # 2. Add search conditions
    search_term = search_params.get('search_term')
    if search_term:
        search_columns = search_params.get('search_columns', [])
        if not search_columns:
            schema_df = con.execute(f"DESCRIBE SELECT * {base_query}").fetchdf()
            search_columns = schema_df[schema_df['column_type'].str.contains('VARCHAR|TEXT', case=False)]['column_name'].tolist()
        
        if search_columns:
            search_or_clauses = []
            for col in search_columns:
                search_or_clauses.append(f'CAST("{col}" AS VARCHAR) ILIKE ?')
                params.append(f'%{search_term}%')
            where_clauses.append(f"({' OR '.join(search_or_clauses)})")

    # --- Build ORDER BY Clause ---
    order_by_clause = ""
    if sort_conditions:
        order_by_parts = [f'"{cond["column"]}" {cond["order"].upper()}' for cond in sort_conditions if cond.get('column') and cond.get('order')]
        if order_by_parts:
            order_by_clause = "ORDER BY " + ", ".join(order_by_parts)

    # --- Construct and Execute Final Query ---
    final_query = f"SELECT * {base_query}"
    if where_clauses:
        final_query += " WHERE " + " AND ".join(where_clauses)
    if order_by_clause:
        final_query += f" {order_by_clause}"

    result_df = con.execute(final_query, params).fetchdf()
    
    # --- Perform the slow file-writing operation ---
    result_df.to_csv(filepath, index=False)
    
    # --- Return the result ---
    # This task does not modify the history state, it overwrites the source file.
    # The session update will be handled by the page reload on the frontend.
    return {
        "message": f"Dataset saved successfully. New row count is {len(result_df)}.",
        "new_row_count": len(result_df)
    }

def export_data_task(history_dir_path, export_format, original_filename, output_dir):
    """
    The background task that generates an export file (CSV, XLSX, etc.)
    and saves it to a designated output directory.
    """
    try:
        # 1. Get the current state of the data
        df_pd = get_current_state_df(history_dir_path)
        if df_pd.empty:
            raise ValueError("Cannot export an empty dataset.")

        # 2. Create a unique filename to prevent collisions
        base_filename = os.path.splitext(original_filename)[0]
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{base_filename}_{unique_id}.{export_format}"
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # 3. Use Pandas to write the file to the specified path
        if export_format == 'csv':
            df_pd.to_csv(output_path, index=False)
        elif export_format == 'tsv':
            df_pd.to_csv(output_path, index=False, sep='\t')
        elif export_format in ['xlsx', 'xls']:
            # xlsxwriter is needed for .xlsx: pip install xlsxwriter
            df_pd.to_excel(output_path, index=False, engine='xlsxwriter')
        elif export_format == 'txt':
            df_pd.to_csv(output_path, index=False, sep=' ')
        else:
            raise ValueError(f"Invalid export format specified: {export_format}")

        # 4. Return the generated filename for the download route
        return {
            "message": f"Successfully created export file: {output_filename}",
            "download_filename": output_filename
        }
    except Exception as e:
        # Re-raise the exception so RQ marks the job as failed
        raise e

def commit_state_to_source_task(history_dir_path, source_filepath):
    """
    The background task that overwrites the original source file
    with the current history state.
    """
    try:
        # 1. Get the current, fully processed DataFrame from the history system
        df_pd = get_current_state_df(history_dir_path)
        if df_pd.empty:
            raise ValueError("Cannot save an empty dataset.")

        # 2. Overwrite the original file with the new data
        # Assumes the source file is a CSV.
        df_pd.to_csv(source_filepath, index=False)
        
        # 3. Return a success message for the user
        return {
            "message": f"Successfully saved all changes to '{os.path.basename(source_filepath)}'.",
            "rows_saved": len(df_pd)
        }
    except Exception as e:
        # Re-raise the exception so RQ marks the job as failed and reports the error
        raise e
    
def save_state_as_new_file_task(history_dir_path, new_filepath):
    """
    The background task that saves the current history state to a new file path.
    """
    try:
        df_pd = get_current_state_df(history_dir_path)
        if df_pd.empty:
            raise ValueError("Cannot save an empty dataset.")

        # Save the current state to the new path
        df_pd.to_csv(new_filepath, index=False)
        
        # Return the path and name for the next step on the frontend
        return {
            "message": "File saved successfully.",
            "new_filepath": new_filepath,
            "new_filename": os.path.basename(new_filepath)
        }
    except Exception as e:
        raise e
