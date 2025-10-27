import os
import re
import json
import math
import uuid
import duckdb
import shutil
import colormap
import logging
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from flask import current_app, session
from sklearn.preprocessing import PowerTransformer
from memory_profiler import memory_usage
from sklearn.preprocessing import PowerTransformer
from pycaret.classification import setup as setup_clf, compare_models as compare_models_clf, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, pull as pull_reg


# ===================================================================
# === STATE MANAGEMENT AND HISTORY
# ===================================================================

def get_org_data_dir():
    """Gets the path to the root shared data directory for the current organization."""
    shared_folder = current_app.config.get('SHARED_DATA_FOLDER', '/app/shared_data')
    org_id = session.get('org_id')
    if not org_id:
        raise Exception("Could not determine organization context for data directory.")
    org_path = os.path.join(shared_folder, str(org_id))
    os.makedirs(org_path, exist_ok=True)
    return org_path

def get_org_datasets_dir():
    """Gets the path to where persistent, shared CSV files are stored for the org."""
    org_path = get_org_data_dir()
    datasets_path = os.path.join(org_path, 'datasets')
    os.makedirs(datasets_path, exist_ok=True)
    return datasets_path

def get_history_dir():
    """
    MODIFIED: Gets the path to the history directory for the current USER and DATASET SESSION.
    This keeps undo/redo history isolated per user, even when working on a shared file.
    e.g., /app/shared_data/{org_id}/sessions/{user_id}/{dataset_session_id}/history/
    """
    org_path = get_org_data_dir()
    user_id = session.get('user', {}).get('id')
    dataset_session_id = session.get('dataset_session_id')

    if not user_id or not dataset_session_id:
        raise Exception("Could not determine user or dataset session context for history.")
        
    session_history_path = os.path.join(org_path, 'sessions', str(user_id), str(dataset_session_id), 'history')
    os.makedirs(session_history_path, exist_ok=True)
    return session_history_path

def get_manifest_path(history_dir_path):
    """Gets the path to the history manifest file FROM A GIVEN PATH."""
    return os.path.join(history_dir_path, 'history_manifest.json')

def read_manifest(history_dir_path):
    """Reads the current history manifest FROM A GIVEN PATH."""
    manifest_path = get_manifest_path(history_dir_path)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {"states": [], "current_state_index": -1}

def write_manifest(manifest, history_dir_path):
    """Writes the updated manifest to a file IN A GIVEN PATH."""
    manifest_path = get_manifest_path(history_dir_path)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

def get_current_state_df(history_dir_path=None):
    """
    Reads the current state's Parquet file.
    If history_dir_path is not provided, it uses the Flask app context.
    """
    if not history_dir_path:
        history_dir_path = get_history_dir()
        
    manifest = read_manifest(history_dir_path)
    if manifest['current_state_index'] == -1:
        return pd.DataFrame()
    
    current_state_file = manifest['states'][manifest['current_state_index']]
    file_path = os.path.join(history_dir_path, current_state_file)
    
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
        
    return pd.DataFrame()

def save_new_state(df, history_dir_path=None, description="Unknown action"): 
    """
    Saves a new DataFrame as a new state in the history.
    """
    is_web_context = not history_dir_path

    if not history_dir_path:
        history_dir_path = get_history_dir()
        
    manifest = read_manifest(history_dir_path)
    
    if manifest['current_state_index'] < len(manifest['states']) - 1:
        states_to_delete = manifest['states'][manifest['current_state_index'] + 1:]
        for state_file in states_to_delete:
            try:
                os.remove(os.path.join(history_dir_path, state_file))
            except OSError:
                pass
        manifest['states'] = manifest['states'][:manifest['current_state_index'] + 1]

    new_state_index = len(manifest['states'])
    new_state_filename = f"state_{new_state_index}.parquet"
    df.to_parquet(os.path.join(history_dir_path, new_state_filename))

    manifest['states'].append(new_state_filename)
    manifest['current_state_index'] = new_state_index
    
    write_manifest(manifest, history_dir_path)

    log_message = (
        f"New state '{new_state_filename}' created. "
        f"Action: '{description}'. "
        f"New shape: ({len(df)}, {len(df.columns)})."
    )
    try:
        # This works when called from a Flask route
        current_app.logger.info(log_message)
    except RuntimeError:
        # This is a fallback for when the function is called from a background task (no app context)
        logging.info(log_message)

    if is_web_context:
        try:
            session['row_count'] = len(df)
            session['col_count'] = len(df.columns)
        except RuntimeError:
            pass



























# ===================================================================
# === DATA ANALYSIS AND FORMATTING
# ===================================================================

def format_bytes(size):
    """Formats file size into KB, MB, GB."""
    if size == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size / p, 2)
    return f"{s} {size_name[i]}"

def get_column_type_summary(schema_df):
    """Categorizes columns into Numeric, Categorical, etc."""
    type_counts = {"Numeric": 0, "Categorical": 0, "Temporal": 0, "Other": 0}
    numeric_types = ('BIGINT', 'DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL')
    temporal_types = ('DATE', 'TIMESTAMP', 'TIME')
    for col_type in schema_df['column_type']:
        if col_type in numeric_types: type_counts["Numeric"] += 1
        elif col_type in temporal_types: type_counts["Temporal"] += 1
        elif col_type == 'VARCHAR': type_counts["Categorical"] += 1
        else: type_counts["Other"] += 1
    return type_counts

def get_column_profiles(df):
    """Generates a list of profile dictionaries for each column from a Pandas DataFrame."""
    total_rows = len(df)
    profiles = []
    for col_name in df.columns:
        missing_count = df[col_name].isnull().sum()
        dtype_str = str(df[col_name].dtype)
        # Create a user-friendly dtype string
        if 'int' in dtype_str: dtype_display = 'INTEGER'
        elif 'float' in dtype_str: dtype_display = 'FLOAT'
        elif 'object' in dtype_str: dtype_display = 'VARCHAR'
        else: dtype_display = dtype_str.upper()

        profile = {
            "name": col_name,
            "dtype": dtype_display,
            "missing_percent": f"{(missing_count / total_rows) * 100:.2f}% Missing" if total_rows > 0 else "0.00% Missing"
        }
        profiles.append(profile)
    return profiles

def generate_full_report_data(filepath: str):
    """Generates a comprehensive data dictionary for the full report."""
    report = {}
    con = duckdb.connect()
    report['header'] = {'dataset_name': os.path.basename(filepath), 'generation_time': datetime.now().strftime("%B %d, %Y, %I:%M %p")}
    schema_df = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{filepath}')").fetchdf()
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{filepath}')").fetchone()[0]
    total_cols = len(schema_df)
    null_counts_sql = " + ".join([f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END)" for col in schema_df['column_name']])
    total_missing = con.execute(f"SELECT {null_counts_sql} FROM read_csv_auto('{filepath}')").fetchone()[0]
    total_cells = total_rows * total_cols
    distinct_rows = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM read_csv_auto('{filepath}')) as t").fetchone()[0]
    report['global_overview'] = {'observations': total_rows, 'variables': total_cols, 'memory_footprint': format_bytes(os.path.getsize(filepath)), 'total_missing': total_missing, 'missing_percent': f"{(total_missing / total_cells) * 100:.2f}" if total_cells > 0 else "0.00", 'duplicate_rows': total_rows - distinct_rows, 'variable_types': get_column_type_summary(schema_df)}
    report['header']['summary'] = f"An analysis of {total_rows:,} rows and {total_cols} columns."
    column_details_list, numeric_column_names = [], []
    for _, row in schema_df.iterrows():
        col_name, col_type = row['column_name'], row['column_type']
        details = {'name': col_name, 'type': col_type}
        quoted_col_name = f'"{col_name}"'
        if col_type in ('BIGINT', 'DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL'):
            numeric_column_names.append(col_name)
            stats_query = f"""SELECT count({quoted_col_name}) as count, avg({quoted_col_name}) as mean, median({quoted_col_name}) as median, stddev_pop({quoted_col_name}) as std_dev, min({quoted_col_name}) as min, max({quoted_col_name}) as max, quantile_disc({quoted_col_name}, 0.25) as q25, quantile_disc({quoted_col_name}, 0.75) as q75, skewness({quoted_col_name}) as skewness, kurtosis({quoted_col_name}) as kurtosis FROM read_csv_auto('{filepath}') WHERE {quoted_col_name} IS NOT NULL"""
            stats_result = con.execute(stats_query).fetchdf()
            details['stats'] = {k: (f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else v) for k, v in stats_result.to_dict('records')[0].items()}
            histogram_dict = con.execute(f"SELECT histogram({quoted_col_name}) FROM read_csv_auto('{filepath}')").fetchone()[0]
            if histogram_dict: details['chart_data'] = {"type": "histogram", "data": [{"x": k, "y": v} for k, v in sorted(histogram_dict.items())]}
        else:
            frequency = con.execute(f"SELECT {quoted_col_name} as category, COUNT(*) as count FROM read_csv_auto('{filepath}') WHERE {quoted_col_name} IS NOT NULL GROUP BY ALL ORDER BY count DESC LIMIT 15").fetchdf().to_dict('records')
            details['stats'] = {'frequency': frequency}
            details['chart_data'] = {"type": "bar", "data": frequency}
        column_details_list.append(details)
    report['column_details'] = column_details_list
    if len(numeric_column_names) > 1:
        try:
            quoted_numeric_cols = ', '.join([f'"{name}"' for name in numeric_column_names])
            polars_df = con.execute(f"SELECT {quoted_numeric_cols} FROM read_csv_auto('{filepath}')").pl()
            corr_matrix = polars_df.corr()
            cmapper = colormap.Color("red", "white", "blue")
            heatmap_data = {"labels": corr_matrix.columns, "rows": []}
            for _, row in enumerate(corr_matrix.to_dicts()):
                formatted_row = []
                for col_name in heatmap_data['labels']:
                    val = row[col_name]
                    if val is None: cell_data = {"value": "N/A", "color": "#f0f0f0"}
                    else: cell_data = {"value": f"{val:.2f}", "color": colormap.rgb2hex(*[int(c * 255) for c in cmapper((val + 1) / 2)])}
                    formatted_row.append(cell_data)
                heatmap_data["rows"].append(formatted_row)
            report['correlation_matrix'] = heatmap_data
        except Exception as e:
            print(f"Could not generate correlation matrix: {e}")
            report['correlation_matrix'] = None
    return report

def get_column_type(schema_df, column_name):
    """Helper to get the type of a specific column from the schema."""
    col_info = schema_df[schema_df['column_name'] == column_name]
    return col_info['column_type'].iloc[0] if not col_info.empty else None

def interpret_correlation(r):
    """Provides a qualitative interpretation of Pearson correlation (r)."""
    val = abs(r)
    if val < 0.2: return "Very Weak"
    elif val < 0.4: return "Weak"
    elif val < 0.6: return "Moderate"
    elif val < 0.8: return "Strong"
    else: return "Very Strong"

def interpret_eta_squared(eta_sq):
    """Provides a qualitative interpretation of Eta-squared (η²)."""
    if eta_sq < 0.01: return "Negligible"
    elif eta_sq < 0.06: return "Weak"
    elif eta_sq < 0.14: return "Moderate"
    else: return "Strong"

def interpret_cramers_v(v):
    """Provides a qualitative interpretation of Cramér's V."""
    if v < 0.1: return "Negligible"
    elif v < 0.2: return "Weak"
    elif v < 0.4: return "Moderate"
    else: return "Strong"


































# ===================================================================
# === SECURE AI CONTEXT & LOGGING
# ===================================================================

def generate_secure_ai_context() -> str:
    """
    Generates a secure, statistics-based context for the AI model.
    This method sends ZERO raw data rows, only aggregated metadata.
    """
    try:
        df = get_current_state_df()
        if df.empty:
            return "Error: The dataset is empty."

        context_parts = ["--- SECURE DATASET CONTEXT ---"]
        
        # Overall Information
        context_parts.append(
            f"### Overall Information\n"
            f"- The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."
        )
        
        missing_total = df.isnull().sum().sum()
        context_parts.append(
            f"- Total missing values: {missing_total:,}."
        )

        # Column Profiles
        context_parts.append("\n### Column Profiles")
        
        for col_name in df.columns:
            col_series = df[col_name]
            col_info = [f"\n---\n\n**Column: `{col_name}`**"]
            
            # Data Type
            dtype_str = str(col_series.dtype)
            if 'int' in dtype_str or 'float' in dtype_str:
                col_type = 'NUMERIC'
            elif 'datetime' in dtype_str:
                col_type = 'DATETIME'
            else:
                col_type = 'VARCHAR'
            col_info.append(f"- Data Type: {col_type}")

            # Missing Values
            missing_count = col_series.isnull().sum()
            missing_percent = (missing_count / df.shape[0]) * 100 if df.shape[0] > 0 else 0
            col_info.append(f"- Missing Values: {missing_count} ({missing_percent:.2f}%)")

            # Type-specific stats
            if col_type == 'NUMERIC' and col_series.notna().any():
                col_info.append(f"- Mean: {col_series.mean():.2f}")
                col_info.append(f"- Median: {col_series.median():.2f}")
                col_info.append(f"- Std Dev: {col_series.std():.2f}")
                col_info.append(f"- Min: {col_series.min()}")
                col_info.append(f"- Max: {col_series.max()}")
            
            elif col_type == 'VARCHAR' and col_series.notna().any():
                cardinality = col_series.nunique()
                col_info.append(f"- Cardinality (Unique Values): {cardinality}")
                
                # Get top 5 most frequent values without revealing sensitive data
                top_5 = col_series.value_counts().nlargest(5)
                top_5_str = ", ".join([f"`{str(k)}` ({v})" for k, v in top_5.items()])
                col_info.append(f"- Top 5 Values: {top_5_str}")

            context_parts.extend(col_info)
            
        return "\n".join(context_parts)
    except Exception as e:
        print(f"CRITICAL Error generating secure AI context: {e}")
        return f"Error: Could not generate dataset context. {e}"
    











































# ===================================================================
# === LOGGING
# ===================================================================

def generate_sql_ai_context() -> str:
    """
    A lightweight, secure context generator for the SQL AI assistant.
    Provides only the dataset's schema information in a robust way.
    """
    try:
        df = get_current_state_df()
        if df.empty:
            return "Error: The dataset is empty."

        detailed_columns = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            dtype_str = str(dtype)
            
            if pd.api.types.is_integer_dtype(dtype):
                dtype_display = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                dtype_display = 'FLOAT'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                dtype_display = 'DATETIME'
            elif pd.api.types.is_bool_dtype(dtype):
                dtype_display = 'BOOLEAN'
            else:
                dtype_display = 'VARCHAR'

            detailed_columns.append({'Column Name': col_name, 'Data Type': dtype_display})

        schema_df = pd.DataFrame(detailed_columns)

        context = f"""
### DATASET SCHEMA
The user wants to query a table named "Dataset".
{schema_df.to_markdown(index=False)}
"""
        return context
        
    except Exception as e:
        import traceback
        print("--- ERROR IN generate_sql_ai_context ---")
        print(traceback.format_exc())
        print("-----------------------------------------")
        return "Error: Could not generate dataset context."

def log_chat_interaction(user_question, ai_answer):
    """Helper function to append a new chat interaction to the log."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'insights_chat_log.json')
        
        logs = []
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r', encoding='utf-8') as f:
                try: 
                    logs = json.load(f)
                except json.JSONDecodeError: 
                    logs = [] # Start fresh if file is corrupt
        
        logs.append({"user": user_question, "ai": ai_answer})

        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error writing to chat log: {e}")

def log_sql_to_history(sql_query, status_message):
    """Helper function to append a new SQL query to the history log."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'sql_log.json')
        
        logs = []
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                try: 
                    logs = json.load(f)
                except json.JSONDecodeError: 
                    logs = []
        
        logs.append({"query": sql_query, "status": status_message})

        with open(log_filepath, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        print(f"Error writing to SQL history log: {e}")

def log_formula_to_history(column_name, expression):
    """Helper function to append a new formula to the history log."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'formula_history.json')
        
        logs = []
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                try: logs = json.load(f)
                except json.JSONDecodeError: logs = []
        
        logs.append({"column": column_name, "expression": expression})

        with open(log_filepath, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error writing to formula history log: {e}")



































# ===================================================================
# === PIPELINE HELPERS
# ===================================================================

def _get_pipeline_creator_schema():
    """Helper specifically for the pipeline creator to get schema."""
    df = get_current_state_df()
    if df.empty:
        return []
    
    schema = []
    for col_name in df.columns:
        col_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col_name]) else 'categorical'
        schema.append({'name': col_name, 'type': col_type})
    return schema

def _generate_default_pipeline_for_creator(filename, schema):
    """Helper to generate the new, generic default pipeline spec for the creator UI."""
    pipeline_steps = []
    step_id_counter = 1

    # Step 1: New - Standardize Column Names
    pipeline_steps.append({"id": step_id_counter, "type": "standardize_column_names", "name": "Standardize Column Names", "config": {"method": "snake_case"}}); step_id_counter += 1
    
    # Step 2: Remove Duplicates (unchanged)
    pipeline_steps.append({"id": step_id_counter, "type": "remove_duplicates", "name": "Remove Duplicates", "config": {"rows": True}}); step_id_counter += 1

    # Step 3: New - Generic Text Cleaning
    pipeline_steps.append({"id": step_id_counter, "type": "clean_text_data", "name": "Clean Text Data", "config": {"trim_whitespace": True, "remove_punctuation": False, "remove_numbers": False, "remove_special_chars": False}}); step_id_counter += 1

    # Step 4: Handle Missing Values (Updated to new generic format)
    pipeline_steps.append({"id": step_id_counter, "type": "handle_missing", "name": "Handle Missing Values", "config": {
        "drop_threshold": 70, 
        "numeric_method": "median", 
        "categorical_method": "mode",
        "datetime_method": "interpolate",
        "numeric_specific_value": 0,
        "categorical_specific_value": "",
        "datetime_specific_value": ""
    }}); step_id_counter += 1

    # Step 5: Handle Outliers (Updated to new generic format)
    pipeline_steps.append({"id": step_id_counter, "type": "handle_outliers", "name": "Handle Outliers", "config": {"method": "iqr", "threshold": 1.5}}); step_id_counter += 1

    return pipeline_steps

def _get_pipelines_dir():
    """Gets the path to the pipelines directory, creating it if it doesn't exist."""
    base_dir = os.path.dirname(session.get('filepath', ''))
    pipelines_path = os.path.join(base_dir, 'pipelines')
    os.makedirs(pipelines_path, exist_ok=True)
    return pipelines_path

def _sanitize_filename(name):
    """Sanitizes a string to be a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip()

def _generate_step_code(step):
    """Translates a single, generic pipeline step object into a Python code string."""
    step_type = step.get('type')
    config = step.get('config', {})
    name = step.get('name', 'Untitled')
    code = f"# Step: {name}\n"

    # --- GENERIC CLEANING & PREP ---
    if step_type == 'remove_duplicates':
        if config.get('rows'):
            code += "df = df.drop_duplicates(ignore_index=True)\n"
            
    elif step_type == 'standardize_column_names':
        method = config.get('method', 'snake_case')
        code += "def to_snake_case(name):\n"
        code += "    s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', name)\n"
        code += "    s2 = re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1).lower()\n"
        code += "    return re.sub(r'[^a-zA-Z0-9_]+', '_', s2).strip('_')\n"
        
        if method == 'snake_case':
            code += "df.columns = [to_snake_case(col) for col in df.columns]\n"
        elif method == 'camelCase':
            code += "snake_cols = [to_snake_case(col) for col in df.columns]\n"
            code += "df.columns = [s.split('_')[0] + ''.join(word.title() for word in s.split('_')[1:]) for s in snake_cols]\n"
        elif method == 'TitleCase':
            code += "df.columns = [to_snake_case(col).replace('_', ' ').title().replace(' ', '') for col in df.columns]\n"
        elif method == 'lowercase':
            code += "df.columns = [col.strip().lower() for col in df.columns]\n"
        elif method == 'uppercase':
            code += "df.columns = [col.strip().upper() for col in df.columns]\n"

    elif step_type == 'clean_text_data':
        code += "categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns\n"
        code += "for col in categorical_cols:\n"
        code += "    if df[col].dtype != 'string': df[col] = df[col].astype(str)\n"
        if config.get('trim_whitespace'):
            code += "    df[col] = df[col].str.strip()\n"
        if config.get('remove_punctuation'):
            code += "    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)\n"
        if config.get('remove_numbers'):
            code += "    df[col] = df[col].str.replace(r'\\d+', '', regex=True)\n"
        if config.get('remove_special_chars'):
            code += "    df[col] = df[col].str.replace(r'[^a-zA-Z0-9\\s]', '', regex=True)\n"

    elif step_type == 'handle_missing':
        # --- START OF REPLACEMENT ---
        drop_threshold = config.get('drop_threshold', 101)
        numeric_method = config.get('numeric_method', 'median')
        categorical_method = config.get('categorical_method', 'mode')
        datetime_method = config.get('datetime_method', 'median')
        
        # 1. Handle dropping columns by threshold
        code += f"drop_perc = {drop_threshold} / 100.0\n"
        code += "df.dropna(axis=1, thresh=len(df) * (1 - drop_perc), inplace=True)\n\n"

        # 2. Handle dropping rows
        code += "numeric_cols = df.select_dtypes(include=np.number).columns\n"
        code += "categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns\n"
        code += "datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns\n"
        code += "cols_to_drop_rows = []\n"
        if numeric_method == 'drop_rows': code += "cols_to_drop_rows.extend(numeric_cols)\n"
        if categorical_method == 'drop_rows': code += "cols_to_drop_rows.extend(categorical_cols)\n"
        if datetime_method == 'drop_rows': code += "cols_to_drop_rows.extend(datetime_cols)\n"
        code += "if cols_to_drop_rows:\n"
        code += "    df.dropna(subset=cols_to_drop_rows, inplace=True)\n\n"

        # 3. Handle imputation for remaining types
        # Numeric
        if numeric_method != 'drop_rows':
            code += "if len(numeric_cols) > 0:\n"
            if numeric_method == 'specific_value':
                val = config.get('numeric_specific_value', 0)
                code += f"    fill_val = pd.to_numeric({repr(val)}, errors='coerce')\n"
                code += "    df[numeric_cols] = df[numeric_cols].fillna(fill_val)\n"
            elif numeric_method in ['mean', 'median']:
                code += f"    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].{numeric_method}())\n"
            elif numeric_method == 'mode':
                code += "    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))\n"
            elif numeric_method in ['ffill', 'bfill']:
                code += f"    df[numeric_cols] = df[numeric_cols].fillna(method='{numeric_method}')\n"
            # --- ADD THIS BLOCK ---
            else:
                code += "    pass # No valid numeric imputation method specified.\n"
            # --- END OF ADDITION ---
        
        # Categorical
        if categorical_method != 'drop_rows':
            code += "\nif len(categorical_cols) > 0:\n"
            if categorical_method == 'specific_value':
                val = config.get('categorical_specific_value', '')
                code += f"    df[categorical_cols] = df[categorical_cols].fillna({repr(val)})\n"
            elif categorical_method == 'mode':
                code += "    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))\n"
            elif categorical_method in ['ffill', 'bfill']:
                code += f"    df[categorical_cols] = df[categorical_cols].fillna(method='{categorical_method}')\n"
            # --- ADD THIS BLOCK ---
            else:
                code += "    pass # No valid categorical imputation method specified.\n"
            # --- END OF ADDITION ---

        # Datetime
        if datetime_method != 'drop_rows':
            code += "\nif len(datetime_cols) > 0:\n"
            if datetime_method == 'specific_value':
                val = config.get('datetime_specific_value', '')
                code += f"    fill_val = pd.to_datetime({repr(val)}, errors='coerce')\n"
                code += "    if pd.notna(fill_val):\n"
                code += "        df[datetime_cols] = df[datetime_cols].fillna(fill_val)\n"
            elif datetime_method in ['mean', 'median']:
                code += f"    for col in datetime_cols:\n"
                code += f"        timestamps = df[col].astype('int64')\n"
                code += f"        fill_val_ts = timestamps.{datetime_method}()\n"
                code += f"        fill_value = pd.to_datetime(fill_val_ts) if pd.notna(fill_val_ts) else pd.NaT\n"
                code += f"        df[col].fillna(fill_value, inplace=True)\n"
            elif datetime_method == 'mode':
                code += "    df[datetime_cols] = df[datetime_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))\n"
            elif datetime_method == 'interpolation':
                code += "    df[datetime_cols] = df[datetime_cols].interpolate(method='time')\n"
            elif datetime_method in ['ffill', 'bfill']:
                code += f"    df[datetime_cols] = df[datetime_cols].fillna(method='{datetime_method}')\n"
            # --- ADD THIS BLOCK ---
            else:
                code += "    pass # No valid datetime imputation method specified.\n"
            # --- END OF ADDITION ---

    # --- GENERIC DATA TRANSFORMATION ---
    elif step_type == 'handle_outliers':
        method = config.get('method', 'iqr')
        threshold = float(config.get('threshold', 1.5))
        code += "numeric_cols = df.select_dtypes(include=np.number).columns\n"
        code += "for col in numeric_cols:\n"
        code += "    col_series = df[col].dropna()\n"
        code += "    if col_series.empty: continue\n"
        if method == 'iqr':
            code += f"    q1, q3 = col_series.quantile(0.25), col_series.quantile(0.75)\n"
            code += f"    iqr = q3 - q1\n"
            code += f"    if iqr > 0:\n"
            code += f"        lower, upper = q1 - ({threshold} * iqr), q3 + ({threshold} * iqr)\n"
            code += f"        df[col] = df[col].clip(lower, upper)\n"
        elif method == 'zscore':
            code += f"    mean, std = col_series.mean(), col_series.std()\n"
            code += f"    if std > 0:\n"
            code += f"        lower, upper = mean - ({threshold} * std), mean + ({threshold} * std)\n"
            code += f"        df[col] = df[col].clip(lower, upper)\n"
        elif method == 'percentile':
            code += f"    lower, upper = col_series.quantile({threshold}), col_series.quantile({1 - threshold})\n"
            code += f"    df[col] = df[col].clip(lower, upper)\n"

    elif step_type == 'transform_skewed_data':
        right_method = config.get('right_skew_method', 'log')
        left_method = config.get('left_skew_method', 'square')
        code += "numeric_cols = df.select_dtypes(include=np.number).columns\n"
        code += "for col in numeric_cols:\n"
        code += "    skewness = df[col].skew()\n"
        code += "    if skewness > 0.5:\n"
        if right_method == 'log': code += "        df[col] = np.log1p(df[col] - df[col].min())\n"
        elif right_method == 'sqrt': code += "        df[col] = np.sqrt(df[col].clip(lower=0))\n"
        elif right_method == 'cbrt': code += "        df[col] = np.cbrt(df[col])\n"
        elif right_method == 'reciprocal': code += "        df[col] = 1 / (df[col] + 1e-6)\n"
        elif right_method == 'box-cox': code += "        if (df[col] > 0).all():\n            pt = PowerTransformer(method='box-cox', standardize=False)\n            df[col] = pt.fit_transform(df[[col]])\n"
        elif right_method == 'yeo-johnson': code += "        pt = PowerTransformer(method='yeo-johnson', standardize=False)\n        df[col] = pt.fit_transform(df[[col]])\n"
        elif right_method == 'arcsinh': code += "        df[col] = np.arcsinh(df[col])\n"
        code += "    elif skewness < -0.5:\n"
        if left_method == 'square': code += "        df[col] = df[col]**2\n"
        elif left_method == 'cube': code += "        df[col] = df[col]**3\n"
        elif left_method == 'tesseractic': code += "        df[col] = df[col]**4\n"
        elif left_method == 'reflection': code += "        df[col] = np.log1p(df[col].max() + 1 - df[col])\n"

    # --- GENERIC PREPROCESSING / ML STEPS ---
    elif step_type == 'feature_scaling':
        method = config.get('method', 'standard')
        scaler_class = {'standard': 'StandardScaler', 'minmax': 'MinMaxScaler', 'robust': 'RobustScaler'}.get(method)
        if scaler_class:
            code += f"from sklearn.preprocessing import {scaler_class}\n"
            code += "numeric_cols = df.select_dtypes(include=np.number).columns\n"
            code += "if len(numeric_cols) > 0:\n"
            code += f"    scaler = {scaler_class}()\n"
            code += "    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n"

    elif step_type == 'categorical_encoding':
        method = config.get('method', 'onehot')
        code += "categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns\n"
        code += "if len(categorical_cols) > 0:\n"
        if method == 'onehot':
            code += "    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)\n"
        elif method == 'hashing':
            code += "    try:\n"
            code += "        from category_encoders.hashing import HashingEncoder\n"
            code += "        encoder = HashingEncoder(n_components=8, cols=categorical_cols)\n"
            code += "        df = encoder.fit_transform(df)\n"
            code += "    except ImportError:\n"
            code += "        print('HashingEncoder requires `category_encoders` library. Skipping step.')\n"
            
    else:
        code += f"pass  # Step type '{step_type}' is not yet implemented.\n"
        
    code += f"print(f'Step \\'{name}\\' executed. Shape: {{df.shape}}')\n"
    return code

def _profiled_pipeline_execution(steps, initial_df):
    """
    A wrapper function designed to be profiled for memory usage.
    It executes a pipeline spec in memory and returns the resulting DataFrame.
    """
    df = initial_df.copy()
    # The actual execution logic is moved into here
    for step in steps:
        executor = PIPELINE_STEP_EXECUTORS.get(step['type'])
        if executor:
            df = executor(step, df)
    return df

def _run_single_benchmark(df, target, problem_type, pipeline_spec):
    """
    Helper to run a PyCaret benchmark, with robust validation for both
    regression and classification tasks.
    """
    # --- 1. Select the correct PyCaret modules based on the problem type ---
    if problem_type == 'regression':
        setup = setup_reg
        compare_models = compare_models_reg
        pull = pull_reg
        default_sort_metric = 'R2'
    else: # Classification
        setup = setup_clf
        compare_models = compare_models_clf
        pull = pull_clf
        default_sort_metric = 'Accuracy'

    # --- 2. Sample the data to a reasonable size for performance ---
    # This prevents benchmarks from taking too long on very large datasets.
    max_cells = 8000
    num_cols = df.shape[1]
    if num_cols > 0:
        max_rows = max_cells // num_cols
        if len(df) > max_rows:
            df_for_benchmark = df.sample(n=max_rows, random_state=123)
        else:
            df_for_benchmark = df
    else:
        df_for_benchmark = df

    # --- 3. CRITICAL: Pre-flight validation of the target column ---
    # This block checks if the data is valid BEFORE sending it to PyCaret.
    if problem_type == 'regression':
        # Check 1: Target must be numeric for regression.
        if not pd.api.types.is_numeric_dtype(df_for_benchmark[target]):
            raise TypeError(
                f"Benchmark failed: The target column '{target}' has a non-numeric data type "
                f"('{df_for_benchmark[target].dtype}'). Regression requires a numeric target. "
                "Check if a pipeline step is changing its type."
            )
        # Check 2: Target must have variance.
        if df_for_benchmark[target].nunique() <= 1:
            raise ValueError(
                f"Benchmark failed: The target column '{target}' has no variance (it contains only one unique value). "
                "A regression model cannot be trained."
            )

    elif problem_type == 'classification':
        # Check 1: Target must have at least two classes.
        if df_for_benchmark[target].nunique() < 2:
            raise ValueError(
                f"Benchmark failed: The target column '{target}' must have at least two unique classes for classification, "
                f"but it only has {df_for_benchmark[target].nunique()}."
            )
        # Check 2: Target should not be a continuous float.
        if pd.api.types.is_float_dtype(df_for_benchmark[target]):
             # We allow floats only if they represent whole numbers (e.g., 1.0, 0.0)
            if not (df_for_benchmark[target].dropna() == df_for_benchmark[target].dropna().astype(int)).all():
                raise TypeError(
                    f"Benchmark failed: The target column '{target}' appears to be a continuous float. "
                    "For classification, please use integer or string labels (e.g., 0, 1 or 'Yes', 'No')."
                )
        # Check 3: Enforce a reasonable cardinality limit to prevent benchmarking on ID columns.
        CARDINALITY_LIMIT = 50 
        if df_for_benchmark[target].nunique() > CARDINALITY_LIMIT:
             raise ValueError(
                f"Benchmark failed: The target column '{target}' has {df_for_benchmark[target].nunique()} unique values, "
                f"which exceeds the limit of {CARDINALITY_LIMIT} for a classification task. This looks more like an ID column."
            )

    # --- 4. Intelligently configure PyCaret based on the user's pipeline ---
    # Avoid double-transforming skewed data if the user already did it.
    has_transform_step = any(step['type'] == 'transform_skewed_data' for step in pipeline_spec.get('steps', []))
    enable_pycaret_transformation = not has_transform_step
    
    # Explicitly define categorical features to prevent PyCaret from misinterpreting them.
    categorical_features = []
    for col in df_for_benchmark.columns:
        if col == target:
            continue
        # Heuristic: numeric columns with low unique values are treated as categorical.
        if pd.api.types.is_numeric_dtype(df_for_benchmark[col]):
            if df_for_benchmark[col].nunique() < 100:
                categorical_features.append(col)
        else: # It's an object/string type
            categorical_features.append(col)
            
    print(f"  - Explicitly defining {len(categorical_features)} categorical features for PyCaret.")
    
    # --- 5. Run the PyCaret benchmark with a final safety net ---
    try:
        setup(data=df_for_benchmark,
              target=target,
              categorical_features=categorical_features,
              normalize=True, 
              transformation=enable_pycaret_transformation,
              remove_multicollinearity=True, 
              verbose=False,
              n_jobs=4)
    except Exception as e:
        # If our checks fail, wrap the complex PyCaret error in a user-friendly message.
        error_message = (
            f"PyCaret's setup function failed. This often happens if the data contains incompatible types "
            f"or if a pipeline step (like scaling or encoding) has created an invalid state. "
            f"Please check the data types and values of your columns after the pipeline has run.\n\n"
            f"Original PyCaret Error: {str(e)}"
        )
        raise Exception(error_message) from e
              
    # --- 6. Compare models and return the results table ---
    compare_models(sort=default_sort_metric)
    return pull()






























# ===================================================================
# === PIPELINE STEP EXECUTORS
# ===================================================================


# Category: Generic Cleaning & Preparation

def _execute_standardize_column_names(step, df):
    config = step.get('config', {})
    method = config.get('method', 'snake_case')
    print(f"  - Standardizing column names to '{method}'.")
    
    def to_snake_case(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        return re.sub(r'[^a-zA-Z0-9_]+', '_', s2).strip('_')

    new_cols = []
    if method == 'snake_case':
        new_cols = [to_snake_case(col) for col in df.columns]
    elif method == 'camelCase':
        new_cols = [to_snake_case(col) for col in df.columns]
        new_cols = [s.split('_')[0] + ''.join(word.title() for word in s.split('_')[1:]) for s in new_cols]
    elif method == 'TitleCase':
        new_cols = [to_snake_case(col).replace('_', ' ').title().replace(' ', '') for col in df.columns]
    elif method == 'lowercase':
        new_cols = [col.strip().lower() for col in df.columns]
    elif method == 'uppercase':
        new_cols = [col.strip().upper() for col in df.columns]
    
    if new_cols:
        df.columns = new_cols
    return df

def _execute_clean_text_data(step, df):
    config = step.get('config', {})
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    print(f"  - Cleaning text data in {len(categorical_cols)} columns.")
    
    for col in categorical_cols:
        # Ensure column is string type before using .str accessor
        if df[col].dtype != 'object' and df[col].dtype != 'string':
             df[col] = df[col].astype(str)
             
        if config.get('trim_whitespace'):
            df[col] = df[col].str.strip()
        if config.get('remove_punctuation'):
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        if config.get('remove_numbers'):
            df[col] = df[col].str.replace(r'\d+', '', regex=True)
        if config.get('remove_special_chars'):
            df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    return df

def _execute_handle_missing(step, df):
    config = step.get('config', {})
    
    # --- START OF REPLACEMENT ---
    drop_threshold = int(config.get('drop_threshold', 101)) / 100.0
    numeric_method = config.get('numeric_method', 'median')
    categorical_method = config.get('categorical_method', 'mode')
    datetime_method = config.get('datetime_method', 'median')

    # 1. Handle dropping columns by missing percentage threshold first
    print(f"  - Dropping columns with >{drop_threshold*100}% missing values.")
    cols_before = set(df.columns)
    df.dropna(axis=1, thresh=len(df) * (1 - drop_threshold), inplace=True)
    cols_after = set(df.columns)
    dropped_cols = list(cols_before - cols_after)
    if dropped_cols:
        print(f"    - Dropped columns: {dropped_cols}")

    # 2. Group all columns that need rows dropped and perform one bulk drop
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns
    
    cols_for_row_drop = []
    if numeric_method == 'drop_rows': cols_for_row_drop.extend(numeric_cols)
    if categorical_method == 'drop_rows': cols_for_row_drop.extend(categorical_cols)
    if datetime_method == 'drop_rows': cols_for_row_drop.extend(datetime_cols)
    
    if cols_for_row_drop:
        print(f"  - Dropping rows with missing values in columns: {cols_for_row_drop}")
        df.dropna(subset=cols_for_row_drop, inplace=True)

    # 3. Impute remaining columns based on their selected method
    # Impute numeric columns
    if numeric_method != 'drop_rows' and len(numeric_cols) > 0:
        print(f"  - Imputing {len(numeric_cols)} numeric columns with method: '{numeric_method}'")
        if numeric_method == 'specific_value':
            fill_value = pd.to_numeric(config.get('numeric_specific_value', 0), errors='coerce')
            df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        elif numeric_method in ['mean', 'median']:
            df[numeric_cols] = df[numeric_cols].fillna(getattr(df[numeric_cols], numeric_method)())
        elif numeric_method in ['mode']:
             df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
        elif numeric_method in ['ffill', 'bfill']:
            df[numeric_cols] = df[numeric_cols].fillna(method=numeric_method)

    # Impute categorical columns
    if categorical_method != 'drop_rows' and len(categorical_cols) > 0:
        print(f"  - Imputing {len(categorical_cols)} categorical columns with method: '{categorical_method}'")
        if categorical_method == 'specific_value':
            df[categorical_cols] = df[categorical_cols].fillna(config.get('categorical_specific_value', ''))
        elif categorical_method == 'mode':
             df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
        elif categorical_method in ['ffill', 'bfill']:
            df[categorical_cols] = df[categorical_cols].fillna(method=categorical_method)
    
    # Impute datetime columns
    if datetime_method != 'drop_rows' and len(datetime_cols) > 0:
        print(f"  - Imputing {len(datetime_cols)} datetime columns with method: '{datetime_method}'")
        if datetime_method == 'specific_value':
            fill_value = pd.to_datetime(config.get('datetime_specific_value'), errors='coerce')
            if not pd.isna(fill_value):
                df[datetime_cols] = df[datetime_cols].fillna(fill_value)
        elif datetime_method in ['mean', 'median']:
            for col in datetime_cols:
                timestamps = df[col].astype('int64')
                fill_val_ts = getattr(timestamps, datetime_method)()
                fill_value = pd.to_datetime(fill_val_ts) if pd.notna(fill_val_ts) else pd.NaT
                df[col].fillna(fill_value, inplace=True)
        elif datetime_method == 'mode':
            df[datetime_cols] = df[datetime_cols].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
        elif datetime_method == 'interpolation':
            df[datetime_cols] = df[datetime_cols].interpolate(method='time')
        elif datetime_method in ['ffill', 'bfill']:
            df[datetime_cols] = df[datetime_cols].fillna(method=datetime_method)
            
    return df


# Category: Data Transformation

def _execute_handle_outliers(step, df):
    config = step.get('config', {})
    method = config.get('method', 'iqr')
    threshold = float(config.get('threshold', 1.5))
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    print(f"  - Clipping outliers in {len(numeric_cols)} numeric columns using method: '{method}'")
    for col in numeric_cols:
        col_series = df[col].dropna()
        if col_series.empty: continue

        if method == 'iqr':
            q1, q3 = col_series.quantile(0.25), col_series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - (threshold * iqr), q3 + (threshold * iqr)
                df[col] = df[col].clip(lower, upper)
        elif method == 'zscore':
            mean, std = col_series.mean(), col_series.std()
            if std > 0:
                lower, upper = mean - (threshold * std), mean + (threshold * std)
                df[col] = df[col].clip(lower, upper)
        elif method == 'percentile':
            lower, upper = col_series.quantile(threshold), col_series.quantile(1 - threshold)
            df[col] = df[col].clip(lower, upper)
            
    return df

def _execute_transform_skewed_data(step, df):
    config = step.get('config', {})
    right_method = config.get('right_skew_method', 'log')
    left_method = config.get('left_skew_method', 'square')
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    print("  - Applying transformations to skewed numeric columns.")
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = df[col].skew()
            
            # Apply transformation for right-skewed data
            if skewness > 0.5:
                print(f"    - Applying '{right_method}' to right-skewed column '{col}' (skew: {skewness:.2f})")
                if right_method == 'log': df[col] = np.log1p(df[col] - df[col].min()) # Shift to be non-negative
                elif right_method == 'sqrt': df[col] = np.sqrt(df[col].clip(lower=0))
                elif right_method == 'cbrt': df[col] = np.cbrt(df[col])
                elif right_method == 'reciprocal': df[col] = 1 / (df[col] + 1e-6)
                elif right_method == 'box-cox':
                    if (df[col] > 0).all():
                        pt = PowerTransformer(method='box-cox', standardize=False)
                        df[col] = pt.fit_transform(df[[col]])
                elif right_method == 'yeo-johnson':
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    df[col] = pt.fit_transform(df[[col]])
                elif right_method == 'arcsinh': df[col] = np.arcsinh(df[col])

            # Apply transformation for left-skewed data
            elif skewness < -0.5:
                print(f"    - Applying '{left_method}' to left-skewed column '{col}' (skew: {skewness:.2f})")
                if left_method == 'square': df[col] = df[col]**2
                elif left_method == 'cube': df[col] = df[col]**3
                elif left_method == 'tesseractic': df[col] = df[col]**4
                elif left_method == 'reflection': df[col] = np.log1p(df[col].max() + 1 - df[col])
                
    return df


# Category: Preprocessing / Machine Learning

def _execute_feature_scaling(step, df):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    config = step.get('config', {})
    method = config.get('method', 'standard')
    scaler_class = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}.get(method)
    
    if not scaler_class:
        print(f"  - Unknown scaling method: {method}. Skipping.")
        return df
        
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        print(f"  - Applying {scaler_class.__name__} to {len(numeric_cols)} numeric columns.")
        scaler = scaler_class()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
    return df

def _execute_categorical_encoding(step, df):
    config = step.get('config', {})
    method = config.get('method', 'onehot')
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    if len(categorical_cols) == 0:
        print("  - No categorical columns found to encode.")
        return df

    print(f"  - Applying {method} encoding to {len(categorical_cols)} columns.")
    if method == 'onehot':
        # drop_first=True to avoid multicollinearity, dummy_na=False to not create a column for NaN
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)
    elif method == 'hashing':
        try:
            from category_encoders.hashing import HashingEncoder
            # n_components=8 is a reasonable default
            encoder = HashingEncoder(n_components=8, cols=categorical_cols)
            df = encoder.fit_transform(df)
        except ImportError:
            print("  - WARNING: 'category_encoders' is not installed. Skipping hashing encoding.")
            
    return df


# Category: Foundational (and Miscellaneous)

def _execute_load_file(step, df):
    print("Executing: Load File")
    return df # The initial dataframe is already loaded


# dictionary that maps the step type to the correct execution function

PIPELINE_STEP_EXECUTORS = {
    "load_file": _execute_load_file,
    "handle_missing": _execute_handle_missing,
    "handle_outliers": _execute_handle_outliers,
    "feature_scaling": _execute_feature_scaling, # This would also need a generic rewrite
    "categorical_encoding": _execute_categorical_encoding, # This would also need a generic rewrite
    "standardize_column_names": _execute_standardize_column_names,
    "clean_text_data": _execute_clean_text_data,
    "transform_skewed_data": _execute_transform_skewed_data,
}














def initialize_new_history_from_df(df):
    """
    Creates a brand new dataset session ID, saves it to the Flask session,
    and initializes a new history directory for it from an existing DataFrame.
    Used after a 'Save As' operation to switch context.
    """
    # 1. Generate a new unique ID for this new dataset context
    new_session_id = str(uuid.uuid4())
    session['dataset_session_id'] = new_session_id

    # 2. Get the new history path (get_history_dir will now use the new session ID)
    history_dir = get_history_dir() # Gets the correct, new, session-specific path
    
    # 3. Clean up any potential old directory at this path (unlikely but safe)
    if os.path.exists(history_dir):
        shutil.rmtree(history_dir)
    os.makedirs(history_dir)
    
    # 4. Save the passed DataFrame as the new initial state
    df.to_parquet(os.path.join(history_dir, 'state_0.parquet'))
    
    # 5. Create a new manifest pointing to this state
    initial_manifest = {
        "states": ["state_0.parquet"], 
        "current_state_index": 0
    }
    write_manifest(initial_manifest, history_dir)