# app/routes.py

from flask import render_template, request, redirect, url_for, current_app, session, g, flash, jsonify, Flask, send_from_directory, send_file, abort, Response
import os
import io
import re
from rq import Queue
import google.generativeai as genai
import duckdb
import polars as pl
from polars import col, sql_expr
from datetime import datetime
import json
import colormap
import math
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import traceback
import psutil
import datetime
import mimetypes
import pyarrow.parquet as pq
import pyarrow as pa
import shutil
import glob
from . import get_supabase
from . import supabase_admin
from functools import wraps
from . import utils
import base64
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
import glob
from werkzeug.utils import secure_filename
from pandas.api.types import is_datetime64_any_dtype
from pandas.tseries.offsets import DateOffset
import uuid

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from category_encoders.hashing import HashingEncoder
from annoy import AnnoyIndex

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score
import polars.selectors as cs

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from scipy import stats


import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pycaret.classification import setup as setup_clf, compare_models as compare_models_clf, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, pull as pull_reg
from memory_profiler import memory_usage

from tasks import run_pipelines_for_comparison_task, run_benchmark_for_comparison_task, execute_pipeline_code_task, smart_impute_task, generate_full_report_task, commit_data_view_task, export_data_task, commit_state_to_source_task, save_state_as_new_file_task


def create_app():
    """Create and configure an instance of the Flask application."""
    
    # --- THE FIX IS HERE ---
    # Add the correct MIME type for .js files.
    # This ensures that module scripts are served with the correct Content-Type header.
    mimetypes.add_type('application/javascript', '.js')
    
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'a-very-secret-key'
    app.config['UPLOAD_FOLDER'] = '../uploads'

    with app.app_context():
        from . import routes

    return app

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Check if the essential session keys exist.
        if 'user' not in session or 'access_token' not in session:
            return redirect(url_for('login', next=request.url))

        try:
            # 2. Prepare the authenticated Supabase client for this specific request.
            # This function will set the user's JWT on the client.
            # The client is then stored in Flask's 'g' object, which is a
            # temporary storage space for the life of a single request.
            get_supabase()
        except Exception as e:
            # If setting the session fails (e.g., expired token), clear the session and force re-login.
            print(f"Error setting Supabase session: {e}")
            session.clear()
            return redirect(url_for('login'))

        # 3. If everything is okay, run the actual route function.
        return f(*args, **kwargs)
    return decorated_function

@current_app.route('/login', methods=['GET', 'POST'])
def login():
    # If a user is already logged in, redirect them to the main app page
    if 'user' in session:
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            # Get a fresh Supabase client instance
            supabase = get_supabase()
            
            # Attempt to sign in the user with their email and password
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            
            # If sign-in is successful, res.user will be populated.
            # Now, fetch the user's profile to get their organization ID and role.
            profile = supabase.table('profiles').select('organization_id, role').eq('id', res.user.id).single().execute()
            
            # Security check: If a user exists in auth but has no profile, something is wrong.
            if not profile.data:
                flash('User profile not found. Please contact support.')
                return render_template('login.html')
            
            org_id = profile.data['organization_id']
            organization = supabase.table('organizations').select('name').eq('id', org_id).single().execute()
            org_name = organization.data['name'] if organization.data else 'Unknown Org'

            # Store all necessary user and session information in the Flask session cookie
            session['user'] = res.user.dict()
            session['org_name'] = org_name
            session['access_token'] = res.session.access_token
            session['refresh_token'] = res.session.refresh_token
            session['org_id'] = profile.data['organization_id']
            session['role'] = profile.data['role']  # Store the user's role

            current_app.logger.info(f"User '{email}' successfully logged in.")
            return redirect(url_for('upload_file'))

        # Inside the except block
        except Exception as e:
            current_app.logger.error(f"Login failed for user '{email}'. Reason: {e}", exc_info=True)
            flash('Invalid credentials or a server error occurred.')

    # If it's a GET request or if login fails, render the login page
    return render_template('login.html')

@current_app.route('/logout')
def logout():
    """Logs the user out by clearing the session and signing out from Supabase."""
    try:
        # Get a stateful client that knows about the current user's token
        supabase = get_supabase()
        
        # Tell Supabase to invalidate the user's token on the server
        supabase.auth.sign_out()
        
    except Exception as e:
        # This might happen if the token is already expired, which is fine.
        # We still want to clear the local session.
        print(f"Error during Supabase sign out (this is often safe to ignore): {e}")
    
    finally:
        # This is the most critical step: clear the Flask session cookie
        session.clear()
        
    # Redirect the user back to the login page
    return redirect(url_for('login'))

@current_app.route('/js_modules/<path:filename>')
def serve_js_module(filename):
    """
    Serves JavaScript module files from the static directory with the correct MIME type.
    This is a specific fix for the "Strict MIME type checking" error.
    """
    # The path is relative to the 'app' directory where the blueprint is.
    # We construct a path to 'app/static/js/cleaner_panels'
    directory = os.path.join(current_app.root_path, 'static', 'js', 'cleaner_panels')
    return send_from_directory(
        directory,
        filename,
        mimetype='application/javascript'
    )



@current_app.route('/data_health_check')
def data_health_check():
    """
    Performs a series of data quality checks on the entire dataset.
    """
    filepath = session.get('filepath')
    if not filepath: return jsonify({"error": "Filepath not found"}), 400

    try:
        df_pd = utils.get_current_state_df()
        if df_pd.empty:
            return jsonify({"error": "No data available to check."})
        df_pl = pl.from_pandas(df_pd)
        
        string_cols = df_pl.select(pl.col(pl.Utf8)).columns
        numeric_cols = df_pl.select(pl.col(pl.NUMERIC_DTYPES)).columns
        total_rows = df_pl.height
        
        checks = {}

        # 1. Missing Values Check
        has_missing_values = df_pl.select(pl.any_horizontal(pl.all().is_null())).to_series().any()
        checks['missing_values'] = {
            "name": "No Missing Values",
            "passed": not has_missing_values,
            "details": "Dataset contains missing (null) cells." if has_missing_values else "All cells are filled."
        }

        # 2. Duplicate Rows Check
        has_duplicates = df_pl.is_duplicated().any()
        checks['duplicate_rows'] = {
            "name": "No Duplicate Rows",
            "passed": not has_duplicates,
            "details": "Exact duplicate rows were found." if has_duplicates else "All rows are unique."
        }

        # 3. Zero Variance Check
        zero_variance_cols = [col.name for col in df_pl if col.drop_nulls().n_unique() <= 1]
        has_zero_variance = len(zero_variance_cols) > 0
        checks['zero_variance'] = {
            "name": "No Constant Columns",
            "passed": not has_zero_variance,
            "details": f"Constant columns: {', '.join(zero_variance_cols)}." if has_zero_variance else "All columns have variance."
        }

        # 4. Global Outlier Detection (IQR Method)
        outlier_cols = []
        if numeric_cols:
            for col_name in numeric_cols:
                q1 = df_pl[col_name].quantile(0.25)
                q3 = df_pl[col_name].quantile(0.75)
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    # Skip check if IQR is zero to avoid flagging all constant values
                    if iqr > 0:
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        if df_pl.select(pl.col(col_name).filter((pl.col(col_name) < lower_bound) | (pl.col(col_name) > upper_bound))).height > 0:
                            outlier_cols.append(col_name)
        
        has_outliers = len(outlier_cols) > 0
        checks['global_outliers'] = {
            "name": "No Extreme Outliers",
            "passed": not has_outliers,
            "details": f"Potential outliers found in: {', '.join(outlier_cols)}." if has_outliers else "Numeric distributions appear to have no extreme outliers."
        }

        # 5. Unwanted Characters Check
        unwanted_char_cols = []
        if string_cols:
            unwanted_regex = r"[^a-zA-Z0-9\s.,!?'\"$%&()-]"
            for col_name in string_cols:
                if df_pl[col_name].drop_nulls().str.contains(unwanted_regex).any():
                    unwanted_char_cols.append(col_name)
        
        has_unwanted_chars = len(unwanted_char_cols) > 0
        checks['unwanted_chars'] = {
            "name": "Standard Characters",
            "passed": not has_unwanted_chars,
            "details": f"Non-standard characters found in: {', '.join(unwanted_char_cols)}." if has_unwanted_chars else "Text values use standard characters."
        }

        # 6. Negative Values Check
        negative_value_cols = []
        if numeric_cols:
            for col_name in numeric_cols:
                if df_pl.select(pl.col(col_name) < 0).sum().item() > 0:
                    negative_value_cols.append(col_name)

        has_negative_values = len(negative_value_cols) > 0
        checks['negative_values'] = {
            "name": "No Negative Values",
            "passed": not has_negative_values,
            "details": f"Negative values found in: {', '.join(negative_value_cols)}." if has_negative_values else "All numeric values are non-negative."
        }
        
        # 7. Whitespace Check
        if string_cols:
            has_whitespace = df_pl.select(pl.any_horizontal(pl.col(string_cols).str.strip_chars() != pl.col(string_cols))).to_series().any()
            checks['whitespace'] = {"name": "No Trailing Whitespace", "passed": not has_whitespace, "details": "Some text values have leading/trailing spaces." if has_whitespace else "Text values are clean."}
        
        # 8. High Cardinality Check
        if string_cols and total_rows > 0:
            high_cardinality_cols = [col for col in string_cols if df_pl[col].drop_nulls().n_unique() / total_rows > 0.5]
            has_high_cardinality = len(high_cardinality_cols) > 0
            checks['high_cardinality'] = {"name": "Low Cardinality", "passed": not has_high_cardinality, "details": f"High cardinality: {', '.join(high_cardinality_cols)}." if has_high_cardinality else "Categorical columns seem reasonable."}
        
        # 9. Mixed Data Type Check
        if string_cols:
            mixed_type_cols = []
            for col_name in string_cols:
                numeric_count = df_pl.select(pl.col(col_name).cast(pl.Float64, strict=False).is_not_null()).sum().item()
                total_non_null = df_pl[col_name].is_not_null().sum()
                if numeric_count > 0 and numeric_count < total_non_null:
                    mixed_type_cols.append(col_name)
            has_mixed_types = len(mixed_type_cols) > 0
            checks['mixed_types'] = { "name": "Consistent Types", "passed": not has_mixed_types, "details": f"Mixed types found in: {', '.join(mixed_type_cols)}." if has_mixed_types else "Column data types appear consistent." }

        # 10. Skewed Distribution Check
        if numeric_cols:
            skewed_cols = []
            skew_df = df_pl.select([pl.col(c).skew().alias(c) for c in numeric_cols]).to_dicts()[0]
            for col_name, skew_value in skew_df.items():
                if skew_value is not None and abs(skew_value) > 1.0:
                    skewed_cols.append(col_name)
            has_skewed_cols = len(skewed_cols) > 0
            checks['skewed_distributions'] = { "name": "Symmetric Distributions", "passed": not has_skewed_cols, "details": f"Highly skewed columns: {', '.join(skewed_cols)}." if has_skewed_cols else "Numeric distributions are reasonably symmetric." }

        return jsonify({"checks": checks})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Health check failed: {str(e)}"}), 500






@current_app.route('/get_undo_redo_status')
def get_undo_redo_status():
    """Checks the manifest and tells the frontend if undo/redo is possible."""
    
    # --- THIS IS THE FIX ---
    # 1. Get the history directory path using the session-aware helper.
    history_dir = utils.get_history_dir()
    # 2. Pass the correct path to the read_manifest function.
    manifest = utils.read_manifest(history_dir)
    # --- END OF FIX ---

    can_undo = manifest['current_state_index'] > 0
    can_redo = manifest['current_state_index'] < len(manifest['states']) - 1
    return jsonify({"can_undo": can_undo, "can_redo": can_redo})

@current_app.route('/undo_change', methods=['POST'])
def undo_change():
    """Moves the state pointer one step back in history."""

    # --- THIS IS THE FIX ---
    history_dir = utils.get_history_dir()
    manifest = utils.read_manifest(history_dir)
    if manifest['current_state_index'] > 0:
        manifest['current_state_index'] -= 1
        utils.write_manifest(manifest, history_dir)
    # --- END OF FIX ---

    return jsonify({"success": True})

@current_app.route('/redo_change', methods=['POST'])
def redo_change():
    """Moves the state pointer one step forward in history."""

    # --- THIS IS THE FIX ---
    history_dir = utils.get_history_dir()
    manifest = utils.read_manifest(history_dir)
    if manifest['current_state_index'] < len(manifest['states']) - 1:
        manifest['current_state_index'] += 1
        utils.write_manifest(manifest, history_dir)
    # --- END OF FIX ---
    
    return jsonify({"success": True})










































@current_app.route('/start_export_job', methods=['POST'])
@login_required
def start_export_job():
    data = request.get_json()
    export_format = data.get('format')
    if not export_format:
        return jsonify({"error": "Export format not specified."}), 400

    try:
        # --- MODIFIED: Get the session-specific history path ---
        history_dir = utils.get_history_dir()
        original_filename = session.get('filename', 'dataset.csv')
        
        org_id = session['org_id']
        output_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id), 'exports')

        # --- Pass the history_dir path to the task ---
        job = current_app.task_queue.enqueue(
            export_data_task,
            args=(history_dir, export_format, original_filename, output_dir),
            job_timeout='15m'
        )
        
        return jsonify({"job_id": job.get_id()}), 202

    except Exception as e:
        return jsonify({"error": f"Failed to start export job: {str(e)}"}), 500


# --- NEW ROUTE 2: The Secure File Downloader ---
@current_app.route('/download_export/<path:filename>')
@login_required
def download_export(filename):
    """Securely serves a generated export file to the user."""
    try:
        org_id = session['org_id']
        # Build the secure path to the exports directory for the user's organization
        export_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id), 'exports')
        
        # Use send_from_directory for security to prevent path traversal attacks
        return send_from_directory(
            export_dir,
            filename,
            as_attachment=True
        )
    except FileNotFoundError:
        abort(404) # If the file doesn't exist, return a 404 error
    except Exception as e:
        flash(f"Error downloading file: {e}")
        return redirect(url_for('data_preview'))


























# ===================================================================
# === FLASK ROUTES
# ===================================================================

def _initialize_dataset_session(df, filename):
    """
    Takes a DataFrame, creates a new unique dataset session, saves it as the
    initial state, and populates the session with relevant metadata.
    """
    try:
        # --- MODIFIED: Use the correct base path for org-level files ---
        # We now get the shared organization directory directly from our new utils helper.
        org_folder = utils.get_org_data_dir()
        # --- END MODIFICATION ---
        
        session['dataset_session_id'] = str(uuid.uuid4())
        current_app.logger.info(f"Initialized new dataset session for file '{filename}'.")

        history_dir = utils.get_history_dir()
        
        row_count = len(df)
        update_metric(session['org_id'], 'rows_processed', row_count)
        
        # Save initial states as Parquet into the new, unique session directory
        raw_path = os.path.join(history_dir, 'raw.parquet')
        df.to_parquet(raw_path)
        df.to_parquet(os.path.join(history_dir, 'state_0.parquet'))
        
        # Create initial manifest in the new session directory
        initial_manifest = {"states": ["state_0.parquet"], "current_state_index": 0}
        utils.write_manifest(initial_manifest, history_dir)
        
        # This part now correctly writes to a subdirectory within the shared volume
        log_files_to_clear = ['insights_chat_log.json', 'formula_history.json', 'sql_log.json']
        for log_file in log_files_to_clear:
            log_filepath = os.path.join(org_folder, log_file) 
            with open(log_filepath, 'w') as f:
                json.dump([], f)

        # This part correctly uses the new datasets subdirectory
        org_datasets_folder = utils.get_org_datasets_dir()
        filepath = os.path.join(org_datasets_folder, filename)
        df.to_csv(filepath, index=False)
        
        # Populate session for the UI
        session['filepath'] = filepath
        session['filename'] = filename
        session['row_count'] = row_count
        session['col_count'] = len(df.columns)
        session['file_size'] = utils.format_bytes(os.path.getsize(filepath))
        
        return True
    except Exception as e:
        # The logger will capture the full traceback for this error
        current_app.logger.error(f"Failed to initialize dataset session: {e}", exc_info=True)
        flash(f"Error initializing dataset session: {e}")
        session.pop('dataset_session_id', None)
        return False


# --- MODIFIED upload_file ROUTE ---
@current_app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    # --- POST request handling for new file uploads ---
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            try:
                # Use secure_filename for safety, though pandas handles the stream
                filename = secure_filename(file.filename)
                initial_df = pd.read_csv(file.stream)
                if _initialize_dataset_session(initial_df, filename):
                    return redirect(url_for('data_preview'))
                else:
                    return redirect(request.url)
            except Exception as e:
                flash(f"Error processing the uploaded file: {e}")
                return redirect(request.url)
    
    # --- GET request handling to display the dashboard ---
    # By default, these are empty
    members = []
    usage = {}
    saved_files = []

    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        os.makedirs(org_folder, exist_ok=True) # Ensure the org's folder exists

        # --- Find all existing CSV datasets for the organization ---
        org_datasets_folder = utils.get_org_datasets_dir()
        search_path = os.path.join(org_datasets_folder, '*.csv')
        for filepath in glob.glob(search_path):
            try:
                file_size = os.path.getsize(filepath)
                saved_files.append({
                    'name': os.path.basename(filepath),
                    'size': utils.format_bytes(file_size)
                })
            except OSError:
                continue # Skip file if it's inaccessible for any reason

        # Sort the files alphabetically by name
        saved_files.sort(key=lambda x: x['name'].lower())

        # If the user is an admin, also fetch the team members and usage data
        if session.get('role') == 'admin':
            # Fetch all members of the organization for display
            profiles_res = supabase_admin.table('profiles').select('id, role').eq('organization_id', org_id).execute()
            
            if profiles_res.data:
                profiles_map = {p['id']: p['role'] for p in profiles_res.data}
                
                # Get all user details from Auth
                all_users_res = supabase_admin.auth.admin.list_users()
                user_list = all_users_res.users if hasattr(all_users_res, 'users') else all_users_res

                # Filter the master user list to get org members
                for user in user_list:
                    if user.id in profiles_map:
                        members.append({'id': user, 'role': profiles_map[user.id]})
                
                # Fetch usage metrics
                supabase = get_supabase()
                metrics_res = supabase.table('usage_metrics').select('*').eq('organization_id', org_id).single().execute()
                if metrics_res.data:
                    usage = metrics_res.data

    except Exception as e:
        flash(f"Could not load dashboard data: {e}", "error")

    # Pass all collected data to the template
    return render_template('upload.html', members=members, usage=usage, saved_files=saved_files)

@current_app.route('/open_file', methods=['POST'])
@login_required
def open_file():
    """Opens an existing dataset and initializes a new session for it."""
    filename_raw = request.form.get('filename')
    if not filename_raw:
        flash("No filename provided.", "error")
        return redirect(url_for('upload_file'))

    try:
        # --- Security: Get the user's specific folder ---
        org_datasets_folder = utils.get_org_datasets_dir()
        filename = secure_filename(filename_raw)
        filepath = os.path.join(org_datasets_folder, filename)

        # Security check logic remains valid, but uses the new base path
        if not os.path.exists(filepath) or not os.path.abspath(filepath).startswith(os.path.abspath(org_datasets_folder)):
            flash(f"File '{filename}' not found or you do not have permission to access it.", "error")
            return redirect(url_for('upload_file'))

        # --- Logic: Read the file and re-use the existing session initializer ---
        df = pd.read_csv(filepath)
        if _initialize_dataset_session(df, filename):
            # If the session is set up correctly, go to the preview page
            return redirect(url_for('data_preview'))
        else:
            # If something went wrong, stay on the upload page
            flash("Could not initialize a session for the selected file.", "error")
            return redirect(url_for('upload_file'))

    except Exception as e:
        flash(f"Error opening file: {e}", "error")
        return redirect(url_for('upload_file'))

@current_app.route('/rename_file', methods=['POST'])
@login_required
def rename_file():
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')

    if not all([old_name, new_name]):
        return jsonify({"error": "Missing file names."}), 400

    try:
        org_datasets_folder = utils.get_org_datasets_dir()
        safe_old_name = secure_filename(old_name)
        safe_new_name = secure_filename(new_name)
        if not safe_new_name.lower().endswith('.csv'):
            safe_new_name += '.csv'

        old_path = os.path.join(org_datasets_folder, safe_old_name)
        new_path = os.path.join(org_datasets_folder, safe_new_name)

        # Security check
        if not os.path.abspath(old_path).startswith(os.path.abspath(org_datasets_folder)):
            abort(403)
        
        if os.path.exists(new_path):
            return jsonify({"error": f"A file named '{safe_new_name}' already exists."}), 409
        
        os.rename(old_path, new_path)
        current_app.logger.info(f"Renamed dataset '{safe_old_name}' to '{safe_new_name}'.")
        return jsonify({"message": f"Successfully renamed to '{safe_new_name}'."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@current_app.route('/delete_file', methods=['POST'])
@login_required
def delete_file():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Filename not provided."}), 400

    try:
        org_datasets_folder = utils.get_org_datasets_dir()
        safe_name = secure_filename(filename)
        file_path = os.path.join(org_datasets_folder, safe_name)

        if not os.path.abspath(file_path).startswith(os.path.abspath(org_datasets_folder)):
            abort(403)

        if os.path.exists(file_path):
            os.remove(file_path)
            current_app.logger.info(f"Deleted dataset '{safe_name}'.")
            # --- MODIFIED: Also remove the associated history directory if it exists ---
            # If the file being deleted is the one in the current session, we can find its history
            if session.get('filename') == filename and session.get('dataset_session_id'):
                session_dir_path = os.path.join(org_datasets_folder, session['dataset_session_id'])
                if os.path.exists(session_dir_path):
                     shutil.rmtree(session_dir_path)

            return jsonify({"message": f"File '{safe_name}' deleted."})
        else:
            return jsonify({"error": "File not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW ROUTE 1: Test Connection ---
@current_app.route('/test_db_connection', methods=['POST'])
@login_required
def test_db_connection():
    data = request.get_json()
    db_type = data.get('db_type')
    user = data.get('user')
    password = data.get('password')
    host = data.get('host')
    port = data.get('port')
    dbname = data.get('dbname')

    if db_type == 'postgresql':
        database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == 'mysql':
        database_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{dbname}"
    else:
        return jsonify({"error": "Unsupported database type"}), 400

    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            return jsonify({"tables": sorted(tables)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'engine' in locals():
            engine.dispose()

# --- NEW ROUTE 2: Import Table ---
@current_app.route('/import_db_table', methods=['POST'])
@login_required
def import_db_table():
    data = request.get_json()
    db_type = data.get('db_type')
    user = data.get('user')
    password = data.get('password')
    host = data.get('host')
    port = data.get('port')
    dbname = data.get('dbname')
    table_name = data.get('table_name')

    if db_type == 'postgresql':
        database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == 'mysql':
        database_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{dbname}"
    else:
        return jsonify({"error": "Unsupported database type"}), 400

    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            
            # --- THIS IS THE FIX ---
            # Instead of inspecting the table, execute a simple, direct query.
            # This avoids the permissions error on the system tables.
            if db_type == 'mysql':
                # MySQL uses backticks for identifiers.
                sql_query = f'SELECT * FROM `{table_name}`'
            else:
                # PostgreSQL (and standard SQL) uses double quotes.
                sql_query = f'SELECT * FROM "{table_name}"'
            
            df = pd.read_sql_query(sql_query, connection)
            # --- END OF FIX ---
        
        # Use a filename based on the table name
        filename = f"{table_name}.csv"
        
        # Call the same helper function as the file upload
        if _initialize_dataset_session(df, filename):
            current_app.logger.info(f"Imported table '{table_name}' from '{db_type}' database '{dbname}'.")
            return jsonify({"message": "Import successful!"}), 200
        else:
            raise Exception("Failed to initialize dataset session from database.")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


@current_app.route('/app_status')
def app_status():
    """API endpoint to get the current application status, like memory usage."""
    try:
        # Get the current Python process
        process = psutil.Process(os.getpid())
        # Get the Resident Set Size (RSS) which is the non-swapped physical memory
        memory_bytes = process.memory_info().rss
        # Reuse our existing helper function to format it nicely
        memory_formatted = utils.format_bytes(memory_bytes)
        
        return jsonify({"memory_usage": memory_formatted})
    except Exception as e:
        return jsonify({"memory_usage": "Error"}), 500

# --- NEW, VIRTUAL-SCROLLING-ENABLED VERSION ---
@current_app.route('/get_table_body', methods=['POST'])
def get_table_body():
    """
    Fetches a specific slice of rows from the CURRENT history state.
    Accepts an `offset` and `limit` for virtual scrolling.
    """
    try:
        # 1. Read the entire current state DataFrame into memory.
        df = utils.get_current_state_df()

        if df.empty:
            return jsonify({"headers": [], "rows": [], "total_rows": 0})

        # 2. Get pagination parameters from the frontend request.
        data = request.get_json() or {}
        offset = data.get('offset', 0)
        limit = data.get('limit', 50) # Fetch 50 rows by default

        # 3. Get the total number of rows for the scrollbar and slice the data.
        total_rows = len(df)
        df_slice = df.iloc[offset : offset + limit]
        
        # 4. Sanitize and prepare the data for JSON response.
        df_sanitized = df_slice.astype(object).where(pd.notna(df_slice), None)
        
        table_rows = df_sanitized.values.tolist()
        headers = df_sanitized.columns.tolist()
        
        # 5. Return the slice, headers, and the TOTAL row count.
        return jsonify({
            "rows": table_rows, 
            "headers": headers,
            "total_rows": total_rows 
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@current_app.route('/get_ag_grid_rows', methods=['POST'])
def get_ag_grid_rows():
    """
    Fetches a slice of rows for AG Grid, now with server-side sorting and filtering.
    """
    try:
        data = request.get_json()
        start_row = data.get('startRow', 0)
        end_row = data.get('endRow', 100)
        sort_model = data.get('sortModel', [])
        filter_model = data.get('filterModel', {})

        df = utils.get_current_state_df()

        # --- 1. APPLY FILTERING (if any) ---
        if filter_model:
            for col, filter_info in filter_model.items():
                # This is a simple example for a 'contains' text filter
                # AG Grid provides much richer filter objects you can parse
                if filter_info['filterType'] == 'text':
                    df = df[df[col].astype(str).str.contains(filter_info['filter'], case=False, na=False)]
                # Example for a 'greaterThan' number filter
                elif filter_info['filterType'] == 'number':
                     df = df[df[col] >= filter_info['filter']]


        # --- 2. APPLY SORTING (if any) ---
        if sort_model:
            sort_cols = [s['colId'] for s in sort_model]
            sort_orders = [s['sort'] == 'asc' for s in sort_model]
            df = df.sort_values(by=sort_cols, ascending=sort_orders)

        total_rows = len(df)
        if df.empty:
            return jsonify({"rows": [], "lastRow": 0})
        
        # --- 3. SLICE THE (now sorted and filtered) DATAFRAME ---
        df_slice = df.iloc[start_row:end_row]
        
        json_string = df_slice.to_json(orient='records', date_format='iso')
        final_json = f'{{"rows": {json_string}, "lastRow": {total_rows}}}'
        
        return Response(final_json, mimetype='application/json')

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- MODIFIED: The /preview route is simpler ---
@current_app.route('/preview')
@login_required
def data_preview():
    # --- THE FIX IS HERE: Read from the current state, not the old CSV ---
    try:
        df = utils.get_current_state_df()
        if df.empty:
            # Handle case where no file has been uploaded yet
            return redirect(url_for('upload_file'))
        column_profiles = utils.get_column_profiles(df)
    except Exception as e:
        flash(f"Error loading data preview: {e}")
        return redirect(url_for('upload_file'))
    
    return render_template('preview.html', column_profiles=column_profiles)

@current_app.route('/full_report')
@login_required
def full_report():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        flash("Please upload a file first to generate a report.")
        return redirect(url_for('upload_file'))
    
    # We no longer pass any data. The template will show a loading message.
    return render_template('full_report.html')


@current_app.route('/start_report_generation', methods=['POST'])
@login_required
def start_report_generation():
    """Dispatches a background job to generate the full report data."""
    try:
        history_dir = utils.get_history_dir()

        low_priority_queue = Queue('low', connection=current_app.redis)
    except Exception as e:
        # This handles cases where a session might be invalid or missing keys
        current_app.logger.error(f"Could not get history directory: {e}", exc_info=True)
        return jsonify({"error": "Could not determine the current data session. Please try reloading the page."}), 40

    try:
        job = low_priority_queue.enqueue(
            generate_full_report_task,
            args=(history_dir,),
            job_timeout='10m' # 5 minute timeout for report generation
        )
        return jsonify({"job_id": job.get_id()}), 202
    except Exception as e:
        return jsonify({"error": f"Failed to start report generation job: {str(e)}"}), 500

@current_app.route('/column_details/<column_name>')
def column_details(column_name):
    """
    Fetches detailed statistics for a specific column from the CURRENT state.
    --- THIS IS THE RESTORED, FULL-FEATURED VERSION ---
    """
    try:
        df = utils.get_current_state_df()
        if column_name not in df.columns:
            return jsonify({"error": "Column not found in current state"}), 404
        
        col_series = df[column_name]
        dtype_str = str(col_series.dtype)

        if 'int' in dtype_str: dtype_display = 'BIGINT'
        elif 'float' in dtype_str: dtype_display = 'DOUBLE'
        elif 'object' in dtype_str: dtype_display = 'VARCHAR'
        elif 'datetime' in dtype_str: dtype_display = 'DATETIME'
        else: dtype_display = dtype_str.upper()
        
        response_data = {"name": column_name, "type": dtype_display}

        if pd.api.types.is_numeric_dtype(col_series):
            stats = col_series.describe().to_dict()
            stats['skew'] = col_series.skew()
            stats['kurtosis'] = col_series.kurtosis()
            stats['sum'] = col_series.sum()
            stats['zeros_%'] = (col_series == 0).mean() * 100
            
            # --- RESTORED METRICS ---
            stats['cardinality'] = col_series.nunique()
            stats['range'] = stats['max'] - stats['min']
            stats['IQR'] = stats['75%'] - stats['25%']
            stats['CV'] = (stats['std'] / stats['mean']) if stats['mean'] != 0 else 0
            # --- END RESTORED METRICS ---

            response_data['stats'] = {k: (f"{v:.3f}" if isinstance(v, (float, np.number)) else v) for k, v in stats.items()}
            hist_data, bin_edges = np.histogram(col_series.dropna(), bins=20)
            response_data['histogram'] = [{"x": float(bin_edges[i]), "y": int(hist_data[i])} for i in range(len(hist_data))]

        else: # Assumed categorical
            stats = {
                'count': int(col_series.count()),
                'cardinality': int(col_series.nunique()),
                'mode': col_series.mode().iloc[0] if not col_series.mode().empty else "N/A"
            }
            
            # --- RESTORED METRICS ---
            str_series = col_series.dropna().astype(str)
            lengths = str_series.str.len()
            stats['min_length'] = int(lengths.min()) if not lengths.empty else 0
            stats['max_length'] = int(lengths.max()) if not lengths.empty else 0
            stats['avg_length'] = lengths.mean() if not lengths.empty else 0
            stats['std_dev_length'] = lengths.std() if not lengths.empty else 0
            # --- END RESTORED METRICS ---
            
            response_data['stats'] = stats
            freq = col_series.value_counts().nlargest(15)
            response_data['frequency_chart_data'] = [{"category": str(k), "count": int(v)} for k, v in freq.items()]
            response_data['full_frequency'] = [{"category": str(k), "count": int(v)} for k, v in col_series.value_counts().items()]

        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- MODIFIED: Returns DISTINCT values and handles filters better ---
@current_app.route('/column_values/<column_name>')
def column_values(column_name):
    """
    API endpoint to get a sorted and filtered list of DISTINCT values from the CURRENT state.
    """
    try:
        # --- THE FIX: Read from the current state ---
        df = utils.get_current_state_df()
        
        if column_name not in df.columns:
            return jsonify({"error": "Column not found"}), 400

        sort_order_str = request.args.get('sort', 'desc').lower()
        ascending = sort_order_str == 'asc'
        filter_str = request.args.get('filter', '')
        
        col_series = df[column_name].dropna()

        if filter_str and pd.api.types.is_numeric_dtype(col_series):
            # Simple numeric filtering for preview
            try:
                if '>' in filter_str: col_series = col_series[col_series > float(filter_str.replace('>', ''))]
                elif '<' in filter_str: col_series = col_series[col_series < float(filter_str.replace('<', ''))]
                elif '-' in filter_str:
                    low, high = map(float, filter_str.split('-'))
                    col_series = col_series[(col_series >= low) & (col_series <= high)]
            except:
                pass # Ignore invalid filters
        
        values = col_series.unique()
        
        # Sort values. Can't sort mixed types so we handle that case.
        try:
            sorted_values = sorted(values, reverse=not ascending)
        except TypeError:
            sorted_values = sorted([str(v) for v in values], reverse=not ascending)

        safe_values = [v.item() if hasattr(v, 'item') else v for v in sorted_values[:1000]]
        
        return jsonify({"values": safe_values})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch values: {str(e)}"}), 500

@current_app.route('/all_relationships/<column_name>')
def all_relationships(column_name):
    """
    Calculates relationship scores from the CURRENT state.
    -- MODIFIED FOR PERFORMANCE using sampling and pre-calculation. --
    """
    try:
        # --- THE FIX: PART 1 - SAMPLING ---
        # For large datasets, running stats on the full data is slow.
        # A large random sample gives statistically equivalent results for this purpose.
        SAMPLE_SIZE = 10000 
        df_pd = utils.get_current_state_df()

        if len(df_pd) > SAMPLE_SIZE:
            df_sample = df_pd.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            df_sample = df_pd
        # --- All subsequent operations will use the much smaller df_sample ---

        if column_name not in df_sample.columns:
            return jsonify({"error": "Column not found"}), 404

        # --- THE FIX: PART 2 - PRE-CALCULATION ---
        # Calculate the entire correlation matrix for all numeric columns at once.
        # This is massively faster than doing it pair-by-pair inside a loop.
        numeric_cols = df_sample.select_dtypes(include=np.number).columns
        if column_name in numeric_cols:
            corr_matrix = df_sample[numeric_cols].corr()
        # --- END OF FIX ---

        all_columns = df_sample.columns.tolist()
        col1_is_numeric = pd.api.types.is_numeric_dtype(df_sample[column_name])
        results = []

        for col2_name in all_columns:
            if col2_name == column_name: continue
            
            col2_is_numeric = pd.api.types.is_numeric_dtype(df_sample[col2_name])
            score, score_name, interpretation, progress_value = 0.0, "N/A", "N/A", 0
            
            if col1_is_numeric and col2_is_numeric:
                # --- THE FIX: PART 3 - LOOKUP INSTEAD OF CALCULATING ---
                # Instead of a slow per-pair calculation, we do a near-instant lookup
                # in the pre-computed matrix.
                correlation = corr_matrix.loc[column_name, col2_name]
                if not np.isnan(correlation):
                    score, score_name, interpretation, progress_value = correlation, "Pearson's r", utils.interpret_correlation(correlation), abs(correlation) * 100
            
            # --- The logic for the other calculations remains the same, but now runs on the SAMPLE ---
            elif not col1_is_numeric and not col2_is_numeric:
                # This now runs on a much smaller DataFrame, making crosstab fast.
                df_temp = df_sample[[column_name, col2_name]].dropna()
                if len(df_temp) > 1 and df_temp[column_name].nunique() > 1 and df_temp[col2_name].nunique() > 1:
                    contingency_table = pd.crosstab(df_temp[column_name], df_temp[col2_name])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n, (r, k) = contingency_table.sum().sum(), contingency_table.shape
                    if min(k - 1, r - 1) == 0: continue
                    cramers_v = np.sqrt((chi2 / n) / min(k - 1, r - 1))
                    score, score_name, interpretation, progress_value = cramers_v, "Cramér's V", utils.interpret_cramers_v(cramers_v), cramers_v * 100
            
            else: # Mixed types (Numeric vs Categorical)
                # This also runs on the much smaller sample.
                df_temp = df_sample[[column_name, col2_name]].dropna()
                if len(df_temp) > 1:
                    numeric_col, categorical_col = (column_name, col2_name) if col1_is_numeric else (col2_name, column_name)
                    if df_temp[categorical_col].nunique() > 1:
                        ss_total = np.sum((df_temp[numeric_col] - df_temp[numeric_col].mean())**2)
                        if ss_total > 0:
                            ss_between = np.sum(df_temp.groupby(categorical_col)[numeric_col].count() * (df_temp.groupby(categorical_col)[numeric_col].mean() - df_temp[numeric_col].mean())**2)
                            eta_squared = ss_between / ss_total
                            score, score_name, interpretation, progress_value = eta_squared, "Eta-squared (η²)", utils.interpret_eta_squared(eta_squared), eta_squared * 100

            results.append({"column_name": col2_name, "score_name": score_name, "score": f"{score:.3f}", "interpretation": interpretation, "progress_value": progress_value})
        
        sorted_results = sorted(results, key=lambda x: x['progress_value'], reverse=True)
        return jsonify(sorted_results)

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Calculation failed: {str(e)}"}), 500

@current_app.route('/edit_cell', methods=['POST'])
@login_required
def edit_cell():
    """
    Receives a single cell update, applies it to the current data state,
    and saves it as a new state in the history.
    """
    data = request.get_json()
    row_index = data.get('row_index')
    column_name = data.get('column_name')
    new_value = data.get('new_value')

    # --- Validation ---
    if row_index is None or column_name is None:
        return jsonify({"error": "Missing row index or column name."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()

        # More validation
        if row_index >= len(df_pd):
            return jsonify({"error": "Row index is out of bounds."}), 400
        if column_name not in df_pd.columns:
            return jsonify({"error": f"Column '{column_name}' not found."}), 404

        # 2. PERFORM the cell update
        original_dtype = df_pd[column_name].dtype

        # --- Smart Type Coercion ---
        # Try to convert the new value to the column's original type for data integrity.
        try:
            if pd.api.types.is_numeric_dtype(original_dtype):
                # For numeric types, use pd.to_numeric which handles various formats
                coerced_value = pd.to_numeric(new_value, errors='raise')
            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                coerced_value = pd.to_datetime(new_value, errors='raise')
            else:
                # For string or other types, just cast to string
                coerced_value = str(new_value)
            
            # Use .at for fast, single-cell access
            df_pd.at[row_index, column_name] = coerced_value

        except (ValueError, TypeError):
             # If coercion fails, return a user-friendly error
             return jsonify({"error": f"Could not convert '{new_value}' to a valid type for this column ({original_dtype})."}), 400


        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_pd, description=f"Edited cell at ({row_index}, {column_name})")
        
        return jsonify({
            "success": True,
            "message": f"Cell at ({row_index}, {column_name}) updated."
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to edit cell: {str(e)}"}), 500
    
@current_app.route('/delete_rows', methods=['POST'])
@login_required
def delete_rows():
    """
    Receives a list of row indices to delete, removes them from the current
    data state, and saves it as a new state in the history.
    """
    data = request.get_json()
    row_indices = data.get('row_indices')

    # --- Validation ---
    if not isinstance(row_indices, list):
        return jsonify({"error": "Invalid payload: row_indices must be a list."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()

        # 2. PERFORM the row deletion using pandas.drop
        # The indices from AG Grid match the DataFrame's integer index.
        df_pd.drop(index=row_indices, inplace=True)
        
        # Reset the index to keep it clean after deletion
        df_pd.reset_index(drop=True, inplace=True)

        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_pd, description=f"Deleted {len(row_indices)} row(s)")
        
        return jsonify({
            "success": True,
            "message": f"{len(row_indices)} row(s) deleted successfully."
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to delete rows: {str(e)}"}), 500
    
@current_app.route('/get_ag_grid_serverside_rows', methods=['POST'])
def get_ag_grid_serverside_rows():
    """
    Handles AG Grid Server-Side Row Model requests. It dynamically builds
    a DuckDB query to handle sorting, filtering, and searching on the server.
    """
    try:
        req = request.get_json()
        start_row = req.get('startRow', 0)
        end_row = req.get('endRow', 100)
        sort_model = req.get('sortModel', [])
        filter_model = req.get('filterModel', {})
        
        # AG Grid's global search text
        quick_filter_text = req.get('quickFilterText', '').strip()

        # Use the current state DataFrame as the source for our query
        current_df = utils.get_current_state_df()
        if current_df.empty:
            return jsonify({"rows": [], "lastRow": 0})

        con = duckdb.connect()
        con.register('current_df', current_df)

        where_clauses = []
        params = []

        # 1. Build WHERE clause from column filters (filterModel)
        for col, filter_info in filter_model.items():
            if filter_info['filterType'] == 'text':
                op = filter_info['type']
                val = filter_info['filter']
                if op == 'contains':
                    where_clauses.append(f'"{col}" ILIKE ?')
                    params.append(f'%{val}%')
                elif op == 'equals':
                    where_clauses.append(f'"{col}" = ?')
                    params.append(val)
            elif filter_info['filterType'] == 'number':
                op = filter_info['type']
                val = filter_info['filter']
                sql_op = {'equals': '=', 'greaterThan': '>', 'lessThan': '<'}.get(op, '=')
                where_clauses.append(f'"{col}" {sql_op} ?')
                params.append(val)
        
        # 2. Build WHERE clause for the global quick filter
        if quick_filter_text:
            # Find all string-like columns to search in
            string_cols = [c for c in current_df.columns if pd.api.types.is_string_dtype(current_df[c])]
            if string_cols:
                search_clauses = []
                for col in string_cols:
                    search_clauses.append(f'CAST("{col}" AS VARCHAR) ILIKE ?')
                    params.append(f'%{quick_filter_text}%')
                where_clauses.append(f"({' OR '.join(search_clauses)})")

        # 3. Build ORDER BY clause from sortModel
        order_by_clause = ""
        if sort_model:
            order_parts = [f'"{s["colId"]}" {s["sort"].upper()}' for s in sort_model]
            order_by_clause = "ORDER BY " + ", ".join(order_parts)

        # Construct the final WHERE clause string
        where_clause_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # --- Query 1: Get the total count of rows AFTER filtering ---
        count_query = f"SELECT COUNT(*) FROM current_df {where_clause_str}"
        total_rows_after_filter = con.execute(count_query, params).fetchone()[0]

        # --- Query 2: Get the slice of data for the current view ---
        limit = end_row - start_row
        offset = start_row
        data_query = f"""
            SELECT * FROM current_df 
            {where_clause_str} 
            {order_by_clause} 
            LIMIT {limit} OFFSET {offset}
        """
        result_df = con.execute(data_query, params).fetchdf()

        # Sanitize data for JSON response
        rows_for_grid = result_df.astype(object).where(pd.notna(result_df), None).to_dict('records')
        
        return jsonify({
            "rows": rows_for_grid,
            "lastRow": total_rows_after_filter
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    



















    








# In app/routes.py, add this new route
@current_app.route('/cleaner')
@login_required
def data_cleaner():
    # We can pass the filename to the cleaner page as well
    filename = session.get('filename', 'No file uploaded')
    return render_template('cleaner.html', filename=filename)

@current_app.route('/get_column_names')
def get_column_names():
    """API to get all column names from the CURRENT state."""
    try:
        df = utils.get_current_state_df()
        columns = df.columns.tolist()
        
        # Match the expected format for the frontend
        renamed_columns = [{'name': name, 'type': str(df[name].dtype)} for name in columns]
        
        return jsonify({"columns": renamed_columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@current_app.route('/smart_impute', methods=['POST'])
def smart_impute():
    """
    Dispatches a background job to perform KNN-style imputation.
    """
    try:
        # Get the history directory path within the web context
        history_dir = utils.get_history_dir()

        high_priority_queue = Queue('high', connection=current_app.redis)
        
        # Enqueue the job, passing the necessary path to the worker
        job = high_priority_queue.enqueue(
            smart_impute_task,
            args=(history_dir,),
            job_timeout='2h' # Imputation can be very long
        )
        
        # Immediately return the job ID
        return jsonify({"job_id": job.get_id()}), 202

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to start Smart-Impute job: {str(e)}"}), 500
    
@current_app.route('/handle_missing_values', methods=['POST'])
def handle_missing_values():
    """
    Handles missing values in a specific column using a selected standard method.
    """
    data = request.get_json()
    column_name = data.get('column_name')
    method = data.get('method')
    specific_value = data.get('value') # For the 'specific_value' method

    if not all([column_name, method]):
        return jsonify({"error": "Missing column name or method."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()

        if column_name not in df_pd.columns:
            return jsonify({"error": f"Column '{column_name}' not found."}), 404

        # --- 2. PERFORM the imputation using Pandas ---
        col_series = df_pd[column_name]
        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        
        # --- Validation ---
        numeric_only_methods = ['mean', 'median']
        if method in numeric_only_methods and not is_numeric:
            return jsonify({"error": f"Method '{method}' can only be applied to numeric columns."}), 400

        # --- Imputation Logic ---
        rows_affected = int(col_series.isnull().sum())
        if rows_affected == 0:
            return jsonify({"message": f"No missing values found in '{column_name}'.", "rows_affected": 0})

        if method == 'mean':
            df_pd[column_name].fillna(col_series.mean(), inplace=True)
        elif method == 'median':
            df_pd[column_name].fillna(col_series.median(), inplace=True)
        elif method == 'mode':
            mode_value = col_series.mode()
            if not mode_value.empty:
                df_pd[column_name].fillna(mode_value.iloc[0], inplace=True)
        elif method == 'ffill': # Forward Fill
            df_pd[column_name].fillna(method='ffill', inplace=True)
        elif method == 'bfill': # Backward Fill
            df_pd[column_name].fillna(method='bfill', inplace=True)
        elif method == 'specific_value':
            if specific_value is None:
                return jsonify({"error": "A value must be provided for this method."}), 400
            try:
                fill_value = pd.to_numeric(specific_value) if is_numeric else str(specific_value)
                df_pd[column_name].fillna(fill_value, inplace=True)
            except ValueError:
                 return jsonify({"error": f"Could not convert '{specific_value}' to a suitable type for this column."}), 400
        elif method == 'drop_rows':
             df_pd.dropna(subset=[column_name], inplace=True)
        else:
            return jsonify({"error": f"Invalid imputation method '{method}'."}), 400

        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_pd, description=f"Applied '{method}' to missing values in '{column_name}'")
        
        return jsonify({
            "message": f"Successfully applied '{method}' to {rows_affected} missing cells in '{column_name}'.",
            "rows_affected": rows_affected
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Imputation failed: {str(e)}"}), 500
    
@current_app.route('/handle_outliers', methods=['POST'])
def handle_outliers():
    """
    Handles outliers using one of three methods, integrated with the state management system.
    """
    data = request.get_json()
    column_name = data.get('column_name')
    definition = data.get('definition')  # "percentile", "iqr", "zscore"
    threshold = data.get('threshold')    # The value from the slider
    action = data.get('action')          # "clip" or "trim"

    if not all([column_name, definition, threshold, action]):
        return jsonify({"error": "Missing parameters in request."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()
        
        # 2. CONVERT to a Polars DataFrame for the operation
        df_pl = pl.from_pandas(df_pd)
        
        # Ensure the column is numeric before proceeding
        if df_pl[column_name].dtype not in pl.NUMERIC_DTYPES:
            return jsonify({"error": f"'{column_name}' is not a numeric column."}), 400

        lower_bound, upper_bound = None, None

        # --- Step 1: Define Outlier Bounds Based on Selected Method ---
        if definition == 'percentile':
            lower_bound = df_pl[column_name].quantile(threshold)
            upper_bound = df_pl[column_name].quantile(1 - threshold)
        elif definition == 'iqr':
            q1 = df_pl[column_name].quantile(0.25)
            q3 = df_pl[column_name].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: # Handle zero IQR case
                 return jsonify({"message": "No outliers found (column has zero variance).", "rows_affected": 0})
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
        elif definition == 'zscore':
            mean = df_pl[column_name].mean()
            std = df_pl[column_name].std()
            if std == 0: # Cannot calculate Z-score if there is no variance
                return jsonify({"message": "No outliers found (column has zero variance).", "rows_affected": 0})
            lower_bound = mean - (threshold * std)
            upper_bound = mean + (threshold * std)
        else:
            return jsonify({"error": "Invalid outlier definition method."}), 400

        if lower_bound is None or upper_bound is None:
            return jsonify({"message": "Could not calculate outlier bounds.", "rows_affected": 0})

        # --- Step 2: Identify Outliers and Apply Action ---
        outlier_mask = (df_pl[column_name] < lower_bound) | (df_pl[column_name] > upper_bound)
        rows_affected = outlier_mask.sum()

        if rows_affected == 0:
            return jsonify({"message": "No outliers found with the selected criteria.", "rows_affected": 0})

        modified_df_pl = None # Initialize variable for the modified DataFrame
        if action == 'clip':
            modified_df_pl = df_pl.with_columns(
                pl.when(outlier_mask)
                  .then(
                      pl.when(pl.col(column_name) < lower_bound)
                        .then(pl.lit(lower_bound))
                        .otherwise(pl.lit(upper_bound))
                  )
                  .otherwise(pl.col(column_name))
                  .alias(column_name)
            )
            message = f"Clipped {rows_affected} outliers in '{column_name}'."
        elif action == 'trim':
            modified_df_pl = df_pl.filter(~outlier_mask)
            message = f"Removed {rows_affected} rows containing outliers in '{column_name}'."
        else:
            return jsonify({"error": "Invalid action specified."}), 400

        # 3. CONVERT the result back to a Pandas DataFrame
        modified_df_pd = modified_df_pl.to_pandas()
        
        # 4. SAVE the modified DataFrame as a new state
        utils.save_new_state(modified_df_pd, description=f"Applied '{action}' to outliers in '{column_name}'")
        
        return jsonify({"message": message, "rows_affected": int(rows_affected)})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Outlier handling failed: {str(e)}"}), 500
    
@current_app.route('/outlier_previews/<column_name>')
def outlier_previews(column_name):
    """
    API endpoint to generate sensitivity analysis data for outlier methods.
    For each method, it calculates the number of outliers at various thresholds.
    """
    # This function no longer needs the original filepath, as it uses the state system.
    
    try:
        # --- THE FIX IS HERE ---
        # 1. Read the CURRENT state of the data, not the original CSV.
        df_pd = utils.get_current_state_df()
        # 2. Convert to a Polars DataFrame for the analysis.
        df_pl = pl.from_pandas(df_pd)
        # 3. Select only the target column and drop nulls for the preview calculations.
        df_pl = df_pl.select(pl.col(column_name)).drop_nulls()
        # --- END OF FIX ---

        if column_name not in df_pl.columns or df_pl[column_name].dtype not in pl.NUMERIC_DTYPES:
            return jsonify({"error": "A valid numeric column must be selected."}), 400

        # --- Prepare data and base stats ---
        col_series = df_pl.get_column(column_name)
        previews = {}

        # --- Method 1: IQR Sensitivity ---
        q1 = col_series.quantile(0.25)
        q3 = col_series.quantile(0.75)
        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            iqr_results = []
            # Generate counts for a range of IQR multipliers
            for multiplier in np.arange(1.0, 3.01, 0.25):
                lower_bound = q1 - (multiplier * iqr)
                upper_bound = q3 + (multiplier * iqr)
                outlier_count = col_series.filter((col_series < lower_bound) | (col_series > upper_bound)).len()
                iqr_results.append({"threshold": round(multiplier, 2), "count": outlier_count})
            previews['iqr'] = iqr_results

        # --- Method 2: Z-score Sensitivity ---
        mean = col_series.mean()
        std = col_series.std()
        if std is not None and std > 0:
            zscore_results = []
            # Generate counts for a range of Z-score thresholds
            for threshold in np.arange(2.0, 4.01, 0.25):
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                outlier_count = col_series.filter((col_series < lower_bound) | (col_series > upper_bound)).len()
                zscore_results.append({"threshold": round(threshold, 2), "count": outlier_count})
            previews['zscore'] = zscore_results

        # --- Method 3: Percentile Sensitivity ---
        percentile_results = []
        # Generate counts for a range of percentile thresholds
        for threshold in [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05]:
            lower_bound = col_series.quantile(threshold)
            upper_bound = col_series.quantile(1 - threshold)
            if lower_bound is not None and upper_bound is not None:
                outlier_count = col_series.filter((col_series < lower_bound) | (col_series > upper_bound)).len()
                percentile_results.append({"threshold": threshold, "count": outlier_count})
        previews['percentile'] = percentile_results
            
        return jsonify({"previews": previews})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate previews: {str(e)}"}), 500
    
@current_app.route('/feature_scaling', methods=['POST'])
def feature_scaling():
    """
    Applies a selected scaling method to a specific numeric column or all
    numeric columns, using the state management system.
    """
    data = request.get_json()
    method = data.get('method')
    target_column = data.get('column_name') # Can be a single column name or None

    if not method:
        return jsonify({"error": "Scaling method not specified."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()

        # 2. CONVERT to a Polars DataFrame for the operation
        df_pl = pl.from_pandas(df_pd)
        
        # --- (Core logic is now performed on df_pl and df_pd) ---
        if target_column:
            # Mode 1: Scale a single, specified column
            if target_column not in df_pl.columns:
                return jsonify({"error": f"Column '{target_column}' not found."}), 404
            # Use cs.numeric() for robust type checking in Polars
            if target_column not in df_pl.select(cs.numeric()).columns:
                return jsonify({"error": f"Column '{target_column}' is not numeric."}), 400
            columns_to_scale = [target_column]
        else:
            # Mode 2: Scale all numeric columns (original behavior)
            columns_to_scale = df_pl.select(cs.numeric()).columns
        
        if not columns_to_scale:
            return jsonify({"message": "No numeric columns found to scale.", "columns_scaled": []})

        df_final_pl = None # Initialize a variable to hold the final polars df

        # Methods that use scikit-learn
        if method in ['standard', 'minmax', 'robust']:
            # This part already uses Pandas, so we can use df_pd directly
            numeric_data_to_scale = df_pd[columns_to_scale]
            
            scaler = None
            if method == 'standard': scaler = StandardScaler()
            elif method == 'minmax': scaler = MinMaxScaler()
            elif method == 'robust': scaler = RobustScaler()
            
            scaled_data = scaler.fit_transform(numeric_data_to_scale)
            df_scaled_pd = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df_pd.index)
            
            df_pd.update(df_scaled_pd)
            # The final result is already in a Pandas DataFrame (df_pd)
            modified_df = df_pd

        # Methods that can be done efficiently with pure Polars
        else:
            scaling_expressions = []
            for col_name in columns_to_scale:
                col_expr = pl.col(col_name)
                
                if method == 'absmax':
                    abs_max = col_expr.abs().max()
                    scaled_expr = pl.when(abs_max == 0).then(0).otherwise(col_expr / abs_max)
                elif method == 'mean_norm':
                    mean = col_expr.mean()
                    col_range = col_expr.max() - col_expr.min()
                    scaled_expr = pl.when(col_range == 0).then(0).otherwise((col_expr - mean) / col_range)
                elif method == 'vector_norm':
                    l2_norm = col_expr.pow(2).sum().sqrt()
                    scaled_expr = pl.when(l2_norm == 0).then(0).otherwise(col_expr / l2_norm)
                else:
                     return jsonify({"error": "Invalid scaler specified."}), 400
                
                scaling_expressions.append(scaled_expr.alias(col_name))
            
            df_final_pl = df_pl.with_columns(scaling_expressions)
            # Convert the final Polars result back to Pandas
            modified_df = df_final_pl.to_pandas()

        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(modified_df, description=f"Applied '{method}' scaling to {len(columns_to_scale)} column(s)")

        return jsonify({
            "message": f"Successfully applied {method.replace('_', ' ').title()} to {len(columns_to_scale)} column(s).",
            "columns_scaled": columns_to_scale
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Feature scaling failed: {str(e)}"}), 500
    
@current_app.route('/categorical_encoding', methods=['POST'])
def categorical_encoding():
    """
    Applies One-Hot or Hashing encoding to a single selected categorical column,
    using the state management system.
    """
    data = request.get_json()
    method = data.get('method')
    target_column = data.get('column_name')

    if not all([method, target_column]):
        return jsonify({"error": "Method or target column not specified."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()
        
        # --- Validation Checks ---
        if target_column not in df_pd.columns:
            return jsonify({"error": f"Column '{target_column}' not found."}), 404
        # Use Pandas' own type checking
        if not pd.api.types.is_string_dtype(df_pd[target_column]):
            return jsonify({"error": f"Column '{target_column}' is not a categorical (text) column."}), 400

        # --- 2. PERFORM the operation using Pandas and Scikit-learn ---
        
        # Isolate the original data that will NOT be encoded or dropped
        df_main = df_pd.drop(columns=[target_column])

        # Isolate the single column to be encoded
        column_to_encode = df_pd[[target_column]]

        if method == 'onehot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
            encoded_data = encoder.fit_transform(column_to_encode)
            
            # Create meaningful new column names
            new_col_names = encoder.get_feature_names_out([target_column])
            df_encoded = pd.DataFrame(encoded_data, columns=new_col_names)
            message_verb = "One-Hot Encoded"

        elif method == 'hashing':
            encoder = HashingEncoder(n_components=8)
            encoded_data = encoder.fit_transform(column_to_encode)
            
            # Rename the new hashed columns to be descriptive
            df_encoded = encoded_data.rename(columns={
                c: f"{target_column}_hash_{c}" for c in encoded_data.columns
            })
            message_verb = "Hash Encoded"

        else:
            return jsonify({"error": "Invalid encoding method."}), 400

        # Concatenate the main data and the new encoded data
        df_final = pd.concat([df_main.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

        # --- 3. SAVE the modified DataFrame as a new state ---
        utils.save_new_state(df_final, description=f"Applied '{method}' categorical encoding to '{target_column}'")

        # Update session info with the new column count for the status bar
        session['col_count'] = len(df_final.columns)
            
        return jsonify({
            "message": f"Successfully {message_verb} column '{target_column}'.",
            "columns_affected": [target_column]
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Encoding failed: {str(e)}"}), 500
    
@current_app.route('/standardize_text_column', methods=['POST'])
@login_required
def standardize_text_column():
    """Applies a specific standardization function to a text column."""
    data = request.get_json()
    column_name = data.get('column_name')
    operation = data.get('operation') # e.g., 'trim_whitespace'

    if not all([column_name, operation]):
        return jsonify({"error": "Missing column name or operation."}), 400

    try:
        df = utils.get_current_state_df()

        if column_name not in df.columns:
            return jsonify({"error": f"Column '{column_name}' not found."}), 404
        
        # Ensure the column is a string type before applying string operations
        if not pd.api.types.is_string_dtype(df[column_name]):
             df[column_name] = df[column_name].astype(str)

        rows_affected = 0 # We can track how many rows were actually changed

        if operation == 'trim_whitespace':
            original_series = df[column_name]
            df[column_name] = df[column_name].str.strip()
            rows_affected = (original_series != df[column_name]).sum()
            op_name = "Trim Whitespace"
        elif operation == 'remove_punctuation':
            df[column_name] = df[column_name].str.replace(r'[^\w\s]', '', regex=True)
            op_name = "Remove Punctuation"
        elif operation == 'remove_numbers':
            df[column_name] = df[column_name].str.replace(r'\d+', '', regex=True)
            op_name = "Remove Numbers"
        elif operation == 'remove_special_chars':
            df[column_name] = df[column_name].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            op_name = "Remove Special Characters"
        else:
            return jsonify({"error": f"Invalid operation '{operation}'."}), 400

        utils.save_new_state(df, description=f"Standardized '{column_name}' using '{operation}' operation.")
        
        message = f"Successfully applied '{op_name}' to '{column_name}'."
        if operation == 'trim_whitespace': # Only easy to calculate for trim
            message += f" {rows_affected} rows were changed."

        return jsonify({"message": message})

    except Exception as e:
        return jsonify({"error": f"Standardization failed: {str(e)}"}), 500
    
# ===================================================================
# === NEW DATETIME CLEANING ROUTES
# ===================================================================

@current_app.route('/format_datetime_column', methods=['POST'])
@login_required
def format_datetime_column():
    """
    Attempts to parse a column into a standardized datetime format.
    Handles various separators, padding, and year formats automatically.
    Now accepts a 'dayfirst' flag to handle DD/MM formats.
    """
    data = request.get_json()
    column_name = data.get('column_name')
    # --- NEW: Get the dayfirst flag from the request, default to False ---
    dayfirst = data.get('dayfirst', False)
    
    if not column_name:
        return jsonify({"error": "Column name not provided."}), 400

    try:
        df = utils.get_current_state_df()
        if column_name not in df.columns:
            return jsonify({"error": f"Column '{column_name}' not found."}), 404

        original_nulls = df[column_name].isnull().sum()
        
        # --- MODIFIED: Pass the dayfirst flag to the function ---
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce', dayfirst=dayfirst)
        
        new_nulls = df[column_name].isnull().sum()
        failed_conversions = new_nulls - original_nulls

        utils.save_new_state(df, description=f"Standardized '{column_name}' to datetime.")

        message = f"Successfully formatted '{column_name}'. "
        if failed_conversions > 0:
            message += f"{failed_conversions} values could not be parsed and were set to null."
        
        return jsonify({"message": message})

    except Exception as e:
        return jsonify({"error": f"Formatting failed: {str(e)}"}), 500

@current_app.route('/handle_datetime_impute', methods=['POST'])
@login_required
def handle_datetime_impute():
    """Imputes missing values in a datetime column using various methods."""
    data = request.get_json()
    column_name, method, value = data.get('column_name'), data.get('method'), data.get('value')
    if not all([column_name, method]):
        return jsonify({"error": "Missing column name or method."}), 400

    try:
        df = utils.get_current_state_df()
        if not is_datetime64_any_dtype(df[column_name]):
            return jsonify({"error": "This operation can only be applied to datetime columns."}), 400

        rows_affected = int(df[column_name].isnull().sum())
        if rows_affected == 0:
            return jsonify({"message": f"No missing values in '{column_name}'."})

        if method in ['mean', 'median']:
            # Convert to numeric (Unix timestamp), calculate, then convert back
            timestamps = df[column_name].astype('int64')
            if method == 'mean':
                fill_val_ts = timestamps.mean()
            else: # median
                fill_val_ts = timestamps.median()
            
            if pd.isna(fill_val_ts): # Handle case where all values are null
                fill_value = pd.NaT
            else:
                fill_value = pd.to_datetime(fill_val_ts)
            df[column_name].fillna(fill_value, inplace=True)

        elif method == 'mode':
            mode_value = df[column_name].mode()
            if not mode_value.empty:
                df[column_name].fillna(mode_value.iloc[0], inplace=True)
        elif method == 'interpolation':
            # Convert the datetime column to a numeric representation (Unix nanoseconds).
            # NaT values will become NaN, which can be interpolated.
            numeric_datetimes = df[column_name].astype('int64')
            # Perform a standard linear interpolation on the numeric values.
            interpolated_numeric = numeric_datetimes.interpolate()
            # Convert the interpolated numeric values back to datetime objects.
            df[column_name] = pd.to_datetime(interpolated_numeric)
        elif method in ['ffill', 'bfill']:
            df[column_name].fillna(method=method, inplace=True)
        elif method == 'specific_value':
            if not value: return jsonify({"error": "A specific value must be provided."}), 400
            fill_value = pd.to_datetime(value, errors='coerce')
            if pd.isna(fill_value): return jsonify({"error": f"Could not parse '{value}' as a valid date."}), 400
            df[column_name].fillna(fill_value, inplace=True)
        elif method == 'drop_rows':
            df.dropna(subset=[column_name], inplace=True)
        else:
            return jsonify({"error": f"Invalid method '{method}'."}), 400

        utils.save_new_state(df, description=f"Imputed missing values in '{column_name}' using '{method}' method.")
        return jsonify({"message": f"Successfully applied '{method}' to {rows_affected} missing cells."})

    except Exception as e:
        return jsonify({"error": f"Imputation failed: {str(e)}"}), 500

@current_app.route('/shift_datetime_column', methods=['POST'])
@login_required
def shift_datetime_column():
    """Shifts a datetime column forward or backward by a specified duration."""
    data = request.get_json()
    column_name = data.get('column_name')
    direction = data.get('direction')
    hours = int(data.get('hours', 0))
    minutes = int(data.get('minutes', 0))

    if not all([column_name, direction]) or direction not in ['forward', 'backward']:
        return jsonify({"error": "Missing or invalid parameters."}), 400

    try:
        df = utils.get_current_state_df()
        if not is_datetime64_any_dtype(df[column_name]):
            return jsonify({"error": "This operation can only be applied to datetime columns."}), 400

        # Create a Timedelta object from the user's input. This is robust.
        time_delta = pd.to_timedelta(f'{hours}h {minutes}m')

        if time_delta == pd.Timedelta(0):
            return jsonify({"message": "No shift applied (duration is zero)."}), 200

        # Apply the shift
        if direction == 'forward':
            df[column_name] = df[column_name] + time_delta
        else: # backward
            df[column_name] = df[column_name] - time_delta

        utils.save_new_state(df, description=f"Shifted '{column_name}' {direction} by {hours}h {minutes}m.")
        
        return jsonify({
            "message": f"Successfully shifted '{column_name}' {direction} by {hours}h {minutes}m."
        })

    except Exception as e:
        return jsonify({"error": f"Time shift operation failed: {str(e)}"}), 500

@current_app.route('/shift_date_column', methods=['POST'])
@login_required
def shift_date_column():
    """Shifts a datetime column by a specified number of days, months, or years."""
    data = request.get_json()
    column_name = data.get('column_name')
    direction = data.get('direction')
    amount = int(data.get('amount', 0))
    unit = data.get('unit') # 'days', 'months', or 'years'

    if not all([column_name, direction, unit]) or direction not in ['forward', 'backward'] or unit not in ['days', 'months', 'years']:
        return jsonify({"error": "Missing or invalid parameters."}), 400

    try:
        df = utils.get_current_state_df()
        if not is_datetime64_any_dtype(df[column_name]):
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            if not is_datetime64_any_dtype(df[column_name]):
                return jsonify({"error": "Column could not be converted to a datetime type."}), 400

        if amount == 0:
            return jsonify({"message": "No shift applied (amount is zero)."}), 200

        # Use DateOffset which correctly handles variable units like months and years (including leap years)
        offset = DateOffset(**{unit: amount})

        # Apply the shift
        if direction == 'forward':
            df[column_name] = df[column_name] + offset
        else: # backward
            df[column_name] = df[column_name] - offset

        utils.save_new_state(df, description=f"Shifted '{column_name}' {direction} by {amount} {unit}.")
        
        return jsonify({
            "message": f"Successfully shifted '{column_name}' {direction} by {amount} {unit}."
        })

    except Exception as e:
        return jsonify({"error": f"Date shift operation failed: {str(e)}"}), 500
    
@current_app.route('/extract_datetime_part', methods=['POST'])
@login_required
def extract_datetime_part():
    """Extracts either the date or time component from a datetime column."""
    data = request.get_json()
    column_name = data.get('column_name')
    part_to_extract = data.get('part_to_extract') # 'date' or 'time'

    if not all([column_name, part_to_extract]) or part_to_extract not in ['date', 'time']:
        return jsonify({"error": "Missing or invalid parameters."}), 400

    try:
        df = utils.get_current_state_df()
        if not is_datetime64_any_dtype(df[column_name]):
            # For robustness, we'll try to convert it first.
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            if not is_datetime64_any_dtype(df[column_name]):
                 return jsonify({"error": "Column could not be converted to a datetime type."}), 400

        # Perform the extraction and immediately format it as a string
        if part_to_extract == 'date':
            # Use strftime to format the date into 'YYYY-MM-DD'
            df[column_name] = df[column_name].dt.strftime('%Y-%m-%d')
        else: # 'time'
            # Use strftime to format the time into 'HH:MM:SS'
            df[column_name] = df[column_name].dt.strftime('%H:%M:%S')

        utils.save_new_state(df, description=f"Extracted {part_to_extract} component from '{column_name}'")
        
        return jsonify({
            "message": f"Successfully extracted the {part_to_extract} component from '{column_name}'."
        })

    except Exception as e:
        return jsonify({"error": f"Extraction operation failed: {str(e)}"}), 500
    
@current_app.route('/column_health_summary')
def column_health_summary():
    """
    API to generate a health summary card for every column in the dataset.
    -- REWRITTEN for maximum robustness and performance using Polars. --
    """
    filepath = session.get('filepath')
    if not filepath: 
        return jsonify({"error": "Filepath not found"}), 400

    try:
        # 1. Load data and convert to a Polars DataFrame. This is the primary source of truth.
        df_pd = utils.get_current_state_df()
        if df_pd.empty:
            return jsonify({"summaries": []})
            
        df_pl = pl.from_pandas(df_pd)
        total_rows = len(df_pl)
        summaries = []

        # 2. Iterate through each column to apply specific, safe logic.
        for col_name in df_pl.columns:
            col_series = df_pl[col_name]
            col_dtype = col_series.dtype
            
            card_data = {
                "name": col_name,
                "type_str": str(col_dtype)
            }
            
            # --- Common Stats (Safe for all types) ---
            missing_count = col_series.is_null().sum()
            card_data['missing_count'] = int(missing_count)
            card_data['missing_percent'] = (missing_count / total_rows) * 100 if total_rows > 0 else 0

            # --- Type-Specific, Robust Calculations ---
            if col_dtype.is_numeric():
                card_data['type'] = 'numeric'
                
                stats = df_pl.select(
                    pl.col(col_name).quantile(0.25).alias("q1"),
                    pl.col(col_name).quantile(0.75).alias("q3"),
                    pl.col(col_name).skew().alias("skewness")
                ).to_dicts()[0]
                
                q1, q3 = stats['q1'], stats['q3']
                
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        outlier_count = col_series.filter(
                            (col_series < lower_bound) | (col_series > upper_bound)
                        ).len()
                        card_data['outlier_count'] = int(outlier_count)
                    else:
                        card_data['outlier_count'] = 0
                else:
                    card_data['outlier_count'] = 0

                card_data['skewness'] = stats['skewness']

            elif col_dtype.is_temporal():
                card_data['type'] = 'datetime'
                
                if col_series.dt.time_zone() is not None:
                    tz_count = col_series.dt.time_zone().drop_nulls().n_unique()
                    card_data['timezone_count'] = int(tz_count)
                else:
                    card_data['timezone_count'] = 0

            elif col_dtype == pl.String:
                card_data['type'] = 'categorical'
                
                # --- THE FIX IS HERE ---
                # Operations inside a .select() must be performed on Polars EXPRESSIONS 
                # (created with pl.col()), not on the Series variable (col_series), 
                # to allow for proper aliasing.
                string_stats = df_pl.select(
                    (pl.col(col_name).str.strip_chars() != pl.col(col_name)).sum().alias("whitespace_count"),
                    pl.col(col_name).n_unique().alias("cardinality")
                ).to_dicts()[0]
                # --- END OF FIX ---
                
                card_data['whitespace_count'] = int(string_stats['whitespace_count'])
                card_data['cardinality'] = int(string_stats['cardinality'])
            
            elif col_dtype.is_boolean():
                card_data['type'] = 'boolean'
                
            else:
                card_data['type'] = 'other'

            summaries.append(card_data)
            
        return jsonify({"summaries": summaries})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500






























@current_app.route('/sql')
@login_required
def sql_view():
    """Renders the SQL Transformations view."""
    filename = session.get('filename', 'No file uploaded')
    # Make sure it renders the renamed template
    return render_template('sql.html', filename=filename)

@current_app.route('/get_schema_details')
@login_required
def get_schema_details():
    """API to get detailed schema info from the CURRENT state."""
    try:
        df = utils.get_current_state_df()
        if df.empty:
            return jsonify({"columns": []})

        total_rows = len(df)
        detailed_columns = []
        for col_name in df.columns:
            dtype_str = str(df[col_name].dtype)
            
            if 'int' in dtype_str: dtype_display = 'BIGINT'
            elif 'float' in dtype_str: dtype_display = 'DOUBLE'
            elif 'object' in dtype_str: dtype_display = 'VARCHAR'
            else: dtype_display = dtype_str.upper()
            
            null_count = df[col_name].isnull().sum()
            is_unique_numpy = total_rows > 0 and df[col_name].nunique() == total_rows and null_count == 0

            # --- THE FIX IS HERE ---
            # Explicitly convert the NumPy boolean to a standard Python boolean.
            # The bool() constructor handles this perfectly.
            detailed_columns.append({
                'name': col_name,
                'type': dtype_display,
                'nullable': bool(null_count > 0), # Also make this explicit for safety
                'is_unique': bool(is_unique_numpy)
            })
            # --- END OF FIX ---
            
        return jsonify({"columns": detailed_columns})
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@current_app.route('/execute_sql', methods=['POST'])
@login_required
def execute_sql():
    """
    Executes a user-provided SQL query against the CURRENT data state.
    """
    data = request.get_json()
    sql_query = data.get('sql')

    if not sql_query:
        return jsonify({"error": "No SQL query provided."}), 400

    try:
        # --- THE FIX IS HERE ---
        # 1. Read the current, most up-to-date data from the state system.
        current_df = utils.get_current_state_df()

        # 2. Register this Pandas DataFrame as a virtual table that DuckDB can query.
        #    The user can now query this virtual table by its registered name ("Dataset").
        con = duckdb.connect()
        con.register('Dataset', current_df)
        
        # 3. Execute the user's query against this virtual table.
        result_df = con.execute(sql_query).fetchdf()
        # --- END OF FIX ---
        
        # Log successful query execution
        status_message = f"Success (Returned {len(result_df)} rows)"
        utils.log_sql_to_history(sql_query, status_message)

        # Sanitize and return the result
        result_df_sanitized = result_df.astype(object).where(pd.notna(result_df), None)
        response_data = {
            "headers": result_df_sanitized.columns.tolist(),
            "rows": result_df_sanitized.values.tolist()
        }
        return jsonify(response_data)

    except Exception as e:
        # Log failed query execution
        utils.log_sql_to_history(sql_query, "Failed")
        
        return jsonify({"error": f"SQL Error: {str(e)}"}), 400
    
@current_app.route('/commit_sql_result', methods=['POST'])
@login_required
def commit_sql_result():
    """
    Executes a SQL query and saves the result as a new state in the history.
    """
    data = request.get_json()
    sql_query = data.get('sql')

    if not sql_query:
        return jsonify({"error": "No SQL query provided."}), 400

    try:
        # 1. Read the current state to query against it
        current_df = utils.get_current_state_df()

        # 2. Execute the query to get the result DataFrame
        con = duckdb.connect()
        con.register('Dataset', current_df)
        result_df = con.execute(sql_query).fetchdf()
        
        if result_df.empty and 'limit 0' not in sql_query.lower():
             if not request.args.get('allow_empty'):
                # A simple safety check. The user might want an empty dataset,
                # but it's often an accident. We could add a confirm step later.
                pass # For now, allow committing empty results.

        # 3. Save the resulting DataFrame as the new state
        utils.save_new_state(result_df, description="Committed SQL Query Result")

        # 4. Log the original query for history
        status_message = f"Success (Committed {len(result_df)} rows as new state)"
        utils.log_sql_to_history(sql_query, status_message)
        current_app.logger.info(f"User committed SQL result. New state has {len(result_df)} rows.")
        return jsonify({"message": f"Successfully committed {len(result_df)} rows..."})

    except Exception as e:
        utils.log_sql_to_history(sql_query, "Commit Failed")
        current_app.logger.error(f"SQL commit failed. Query: {sql_query}. Error: {e}", exc_info=True)
        return jsonify({"error": f"SQL Commit Error: {str(e)}"}), 400
    
@current_app.route('/generate_sql_query', methods=['POST'])
@login_required
def generate_sql_query():
    """
    Uses AI to convert a natural language prompt into a SQL query string.
    This version receives the schema context directly from the front-end.
    """
    # 1. Configure AI Model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable not set."}), 500
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return jsonify({"error": f"Failed to configure AI model: {e}"}), 500

    # 2. Get data from the request body
    request_data = request.get_json()
    user_prompt = request_data.get('prompt')
    dataset_context = request_data.get('context')

    if not user_prompt or not dataset_context:
        return jsonify({"error": "Missing prompt or dataset context."}), 400

    try:
        # 3. Assemble the prompt
        prompt = f"""
You are an expert SQL developer who translates natural language requests into a single, valid DuckDB SQL query.

**CRITICAL RULES:**
1.  Your response MUST be ONLY the SQL query itself, with no explanations or markdown formatting.
2.  If you cannot create a valid query from the request, you MUST return the single character `0`.
3.  You MUST use column names exactly as they appear in the schema, enclosing them in double quotes (e.g., "SalePrice").
4.  The ONLY valid table name you can query is "Dataset".

---
**CONTEXT:**
{dataset_context}
---

**USER REQUEST:** "{user_prompt}"

Based on all the rules and context, generate the SQL query now.
"""

        # 4. Call the AI
        response = model.generate_content(prompt)
        sql_query = response.text.strip().replace('`sql', '').replace('`', '')

        # --- THE LOGGING BLOCK HAS BEEN REMOVED FROM HERE ---

        # 5. Return the result to the front-end
        return jsonify({"sql_query": sql_query})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during AI processing: {str(e)}"}), 500
    
@current_app.route('/get_sql_history')
@login_required
def get_sql_history():
    """API endpoint to fetch the contents of the SQL history log file."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'sql_log.json')
        
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                logs = json.load(f)
                return jsonify(logs)
        else:
            return jsonify([])
    except Exception as e:
        print(f"Error reading SQL history log: {e}")
        return jsonify({"error": str(e)}), 500

































@current_app.route('/visualizations')
@login_required
def visualizations_view():
    """Renders the Visualizations dashboard view."""
    filename = session.get('filename', 'No file uploaded')
    return render_template('visualizations.html', filename=filename)

@current_app.route('/get_all_univariate_charts/<column_name>')
def get_all_univariate_charts(column_name):
    """
    Calculates and returns all data needed for all univariate numerical charts
    from the CURRENT data state.
    """
    try:
        # --- THE FIX: Read from the current state ---
        df = utils.get_current_state_df()
        
        if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            return jsonify({"error": "A valid numeric column must be selected from the current dataset."})

        col_series = df[column_name].dropna()
        if col_series.empty:
            return jsonify({"error": "Column contains no valid data to plot."})

        response_data = {}

        # 1. Data for Histogram
        hist_data, bin_edges = np.histogram(col_series, bins=20)
        response_data['histogram'] = [{"x": float(bin_edges[i]), "y": int(hist_data[i])} for i in range(len(hist_data))]
        
        # 2. Data for CDF Plot
        sorted_data = np.sort(col_series)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        response_data['cdf'] = [{"x": float(val), "y": float(prob)} for val, prob in zip(sorted_data, cdf)]

        # 3. Data for Strip/Jitter Plot
        sample_size = min(1000, len(col_series))
        response_data['strip_plot'] = col_series.sample(n=sample_size).tolist()

        # 4. Data for Q-Q Plot
        probplot_results = stats.probplot(col_series, dist="norm")
        theoretical_quantiles = probplot_results[0][0].tolist()
        sample_quantiles = probplot_results[0][1].tolist()
        response_data['qq_plot'] = [{"x": x, "y": y} for x, y in zip(theoretical_quantiles, sample_quantiles)]

        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate chart data: {str(e)}"}), 500
    
@current_app.route('/get_bivariate_data', methods=['POST'])
def get_bivariate_data():
    """
    Fetches data for bivariate charts from the CURRENT data state.
    """
    data = request.get_json()
    col1_name = data.get('col1')
    col2_name = data.get('col2')
    col3_name = data.get('col3', None)

    if not all([col1_name, col2_name]):
        return jsonify({"error": "Two columns must be specified."}), 400

    try:
        # --- THE FIX: Read from the current state ---
        df = utils.get_current_state_df()
        
        col1_is_numeric = pd.api.types.is_numeric_dtype(df[col1_name])
        col2_is_numeric = pd.api.types.is_numeric_dtype(df[col2_name])
        
        response_data = {}

        # Case 1: Numeric vs Numeric
        if col1_is_numeric and col2_is_numeric:
            cols_to_use = [col1_name, col2_name]
            if col3_name and pd.api.types.is_numeric_dtype(df[col3_name]):
                cols_to_use.append(col3_name)
            
            df_sample = df[cols_to_use].dropna().sample(n=min(2000, len(df.dropna())))
            
            if col3_name in df_sample.columns: # Bubble chart data
                response_data['bubble_data'] = df_sample.to_dict('records')
            else: # Scatter plot data
                response_data['scatter_data'] = df_sample.to_dict('records')

        # Case 2: Categorical vs Categorical
        elif not col1_is_numeric and not col2_is_numeric:
            # Group by and count using Pandas
            stacked_bar_df = df.groupby([col1_name, col2_name]).size().reset_index(name='count')
            response_data['stacked_bar_data'] = stacked_bar_df.to_dict('records')
        else:
            return jsonify({"error": "Unsupported column type combination for bivariate analysis."})

        return jsonify(response_data)

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate bivariate data: {str(e)}"}), 500

@current_app.route('/get_visualization_data/<column_name>')
def get_visualization_data(column_name):
    """
    A dedicated endpoint to fetch basic data for the Visualizations page from the CURRENT data state.
    """
    try:
        # --- THE FIX: Read from the current state ---
        df = utils.get_current_state_df()
        
        if column_name not in df.columns:
            return jsonify({"error": "Column not found"}), 404
            
        col_series = df[column_name]
        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        
        response_data = {"name": column_name, "type": "numeric" if is_numeric else "categorical"}

        if is_numeric:
            hist_data, bin_edges = np.histogram(col_series.dropna(), bins=20)
            response_data['histogram'] = [{"x": float(bin_edges[i]), "y": int(hist_data[i])} for i in range(len(hist_data))]
        else: # Assumed categorical
            freq = col_series.dropna().value_counts().nlargest(15)
            response_data['frequency_chart_data'] = [{"category": str(k), "count": int(v)} for k, v in freq.items()]
            
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch visualization data: {str(e)}"}), 500
    
# --- NEW ROUTE FOR PYTHON-GENERATED CHARTS ---
@current_app.route('/generate_python_chart', methods=['POST'])
@login_required
def generate_python_chart():
    """
    Generates a single static chart image on the server using Matplotlib/Seaborn.
    This is the final, complete version with all plot types, themes, and fixes.
    """
    # 1. Get data from the frontend request
    data = request.get_json()
    chart_type = data.get('chart_type')
    x_col = data.get('x_col')
    theme = data.get('theme', 'native') # Default to 'native' if not provided

    # 2. Basic validation
    if not chart_type or not x_col:
        return jsonify({"error": "Chart type and X-axis column must be specified."}), 400

    fig = None # Initialize figure variable to ensure it's always defined for the finally block
    try:
        # 3. Load the current dataset state
        df = utils.get_current_state_df()
        if x_col not in df.columns:
            return jsonify({"error": f"Column '{x_col}' not found."}), 404

        # 4. Find helper columns required for bivariate and multivariate plots
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Find the first available numeric/categorical column that is NOT the target column
        other_numeric_col = next((col for col in numeric_cols if col != x_col), None)
        other_categorical_col = next((col for col in categorical_cols if col != x_col), None)

        # 5. Define theme dictionaries
        native_theme_rc = {
            "axes.facecolor": "#2C2541", "figure.facecolor": "#161220", "grid.color": "#4a3f6e", 
            "axes.labelcolor": "#a095c0", "xtick.color": "#a095c0", "ytick.color": "#a095c0", 
            "text.color": "#E0D9F0", "patch.edgecolor": "#161220"
        }
        white_theme_rc = {
            "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF", "grid.color": "#EAEAEA",
            "axes.labelcolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
            "text.color": "#000000", "patch.edgecolor": "#FFFFFF"
        }

        # 6. Apply the selected theme
        if theme == 'white':
            sns.set_theme(style="whitegrid", rc=white_theme_rc)
            title_color = 'black'
        else: # Default to native theme
            sns.set_theme(style="whitegrid", rc=native_theme_rc)
            title_color = 'white'
        
        # 7. Charting Logic: Handle figure-level vs. axes-level plots
        
        # Figure-level plots (jointplot, pairplot) are handled differently
        if chart_type in ['jointplot', 'pairplot']:
            if chart_type == 'jointplot' and other_numeric_col:
                g = sns.jointplot(data=df, x=x_col, y=other_numeric_col, kind="scatter", color="#9D4EDD")
                fig = g.fig
            elif chart_type == 'pairplot' and len(numeric_cols) > 1:
                df_sample = df.sample(n=min(len(df), 500))
                g = sns.pairplot(df_sample[numeric_cols], corner=True)
                fig = g.fig
        
        # Axes-level plots (most other plots)
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # --- NUMERIC PLOTS ---
            if chart_type == 'violin': sns.violinplot(data=df, y=x_col, ax=ax, color="#9D4EDD", cut=0)
            elif chart_type == 'boxplot': sns.boxplot(data=df, y=x_col, ax=ax, color="#9D4EDD")
            elif chart_type == 'displot': sns.histplot(data=df, x=x_col, kde=True, ax=ax, color="#9D4EDD")
            elif chart_type == 'kdeplot': sns.kdeplot(data=df, x=x_col, fill=True, ax=ax, color="#9D4EDD")
            elif chart_type == 'ecdfplot': sns.ecdfplot(data=df, x=x_col, ax=ax, color="#9D4EDD")
            elif chart_type == 'rugplot': sns.rugplot(data=df, x=x_col, ax=ax, color="#9D4EDD"); sns.kdeplot(data=df, x=x_col, ax=ax, color="#4a3f6e")
            elif chart_type in ['scatterplot', 'hexbin', 'kde2d', 'lineplot', 'regplot']:
                if other_numeric_col:
                    df_clean = df[[x_col, other_numeric_col]].dropna()
                    if chart_type == 'scatterplot': sns.scatterplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'hexbin': ax.hexbin(df_clean[x_col], df_clean[other_numeric_col], gridsize=20, cmap='viridis')
                    elif chart_type == 'kde2d': sns.kdeplot(data=df_clean, x=x_col, y=other_numeric_col, fill=True, ax=ax)
                    elif chart_type == 'lineplot': sns.lineplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'regplot': sns.regplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                else: ax.text(0.5, 0.5, 'Requires a second numeric column', ha='center', va='center', color=title_color)
            elif chart_type in ['stripplot', 'swarmplot', 'pointplot', 'barplot'] and df[x_col].dtype in [np.number]:
                if other_categorical_col:
                    df_sample = df.sample(n=min(len(df), 1000))
                    if chart_type == 'stripplot': sns.stripplot(data=df_sample, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'swarmplot': sns.swarmplot(data=df_sample, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD", size=3)
                    elif chart_type == 'pointplot': sns.pointplot(data=df, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'barplot': sns.barplot(data=df, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                else: ax.text(0.5, 0.5, 'Requires a categorical column', ha='center', va='center', color=title_color)
            elif chart_type == 'heatmap' and df[x_col].dtype in [np.number]:
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 6})
                else: ax.text(0.5, 0.5, 'Requires multiple numeric columns', ha='center', va='center', color=title_color)

            # --- CATEGORICAL PLOTS ---
            elif chart_type == 'countplot':
                order = df[x_col].value_counts().index[:15]
                sns.countplot(data=df, y=x_col, ax=ax, color="#9D4EDD", order=order)
            elif chart_type == 'pieplot':
                counts = df[x_col].value_counts()
                top_n = counts.nlargest(7)
                if len(counts) > 7: top_n['Other'] = counts.iloc[7:].sum()
                pie_labels = top_n.index
                pie_colors = sns.color_palette("viridis", len(top_n))
                ax.pie(top_n, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors, textprops={'color': title_color})
            elif chart_type in ['cat_boxplot', 'cat_violinplot', 'cat_barplot']:
                if other_numeric_col:
                    order = df[x_col].value_counts().index[:15]
                    if chart_type == 'cat_boxplot': sns.boxplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order)
                    elif chart_type == 'cat_violinplot': sns.violinplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order, cut=0)
                    elif chart_type == 'cat_barplot': sns.barplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order)
                else: ax.text(0.5, 0.5, 'Requires a numeric column', ha='center', va='center', color=title_color)
            elif chart_type in ['grouped_countplot', 'cat_heatmap']:
                if other_categorical_col:
                    if chart_type == 'grouped_countplot':
                        top_x_cats = df[x_col].value_counts().index[:10]
                        top_hue_cats = df[other_categorical_col].value_counts().index[:5]
                        df_filtered = df[df[x_col].isin(top_x_cats) & df[other_categorical_col].isin(top_hue_cats)]
                        sns.countplot(data=df_filtered, y=x_col, hue=other_categorical_col, ax=ax, order=top_x_cats)
                    elif chart_type == 'cat_heatmap':
                        contingency_table = pd.crosstab(df[x_col], df[other_categorical_col])
                        
                        # FIX 1: Reduce the number of columns to a more reasonable number like 8
                        top_cols = contingency_table.sum().sort_values(ascending=False).index[:8]
                        contingency_table_sliced = contingency_table[top_cols]
                        
                        # FIX 2: Dynamically decide whether to show annotations
                        # If the table is still too big, don't show numbers to avoid overlap.
                        show_annotations = True
                        if contingency_table_sliced.shape[0] > 10 or contingency_table_sliced.shape[1] > 10:
                            show_annotations = False

                        sns.heatmap(
                            contingency_table_sliced, 
                            ax=ax, 
                            annot=show_annotations, # Use the dynamic boolean
                            fmt='d', 
                            cmap='viridis',
                            annot_kws={"size": 8} # FIX 3: Make the font size smaller
                        )
                        ax.tick_params(axis='x', rotation=90)
                else: ax.text(0.5, 0.5, 'Requires a second categorical column', ha='center', va='center', color=title_color)
            
            else:
                 ax.text(0.5, 0.5, 'Plot not applicable or data unavailable', ha='center', va='center', color=title_color)

            ax.set_title(chart_type.replace('plot', ' Plot').title(), color=title_color, fontsize=10)

        # 8. Convert plot to an in-memory image and return it
        if fig:
            plt.tight_layout()
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            return Response(img_buffer.getvalue(), mimetype='image/png')
        else:
            return jsonify({"error": "Could not generate a figure for this plot type with the current data."}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate chart: {str(e)}"}), 500
    
    finally:
        # 9. CRITICAL: Close the figure to free up server memory
        if fig:
            plt.close(fig)
            
@current_app.route('/generate_all_python_charts', methods=['POST'])
@login_required
def generate_all_python_charts():
    """
    Generates a batch of relevant static charts for a given column
    and returns them as Base64-encoded images in a single JSON object.
    """
    data = request.get_json()
    column_name = data.get('column_name')
    column_type = data.get('column_type')

    if not all([column_name, column_type]):
        return jsonify({"error": "Column name and type must be specified."}), 400

    df = utils.get_current_state_df()
    if column_name not in df.columns:
        return jsonify({"error": f"Column '{column_name}' not found."}), 404
        
    charts_to_generate = []
    if column_type == 'numeric':
        # --- EXPANDED LIST OF 18 NUMERIC PLOTS ---
        charts_to_generate = [
            # Original Univariate
            {'type': 'violin', 'title': f'Violin Plot'}, {'type': 'boxplot', 'title': f'Box Plot'}, {'type': 'displot', 'title': f'Distribution Plot'},
            # New Univariate
            {'type': 'kdeplot', 'title': 'KDE Plot'}, {'type': 'ecdfplot', 'title': 'ECDF Plot'}, {'type': 'rugplot', 'title': 'Rug Plot'},
            # Bivariate (requires a second column)
            {'type': 'scatterplot', 'title': 'Scatter Plot'}, {'type': 'jointplot', 'title': 'Joint Plot'}, {'type': 'hexbin', 'title': 'Hexbin Density Plot'},
            {'type': 'kde2d', 'title': '2D KDE Plot'}, {'type': 'lineplot', 'title': 'Line Plot'}, {'type': 'regplot', 'title': 'Regression Plot'},
            # Categorical vs Numeric (requires a categorical column)
            {'type': 'stripplot', 'title': 'Strip Plot'}, {'type': 'swarmplot', 'title': 'Swarm Plot'}, {'type': 'pointplot', 'title': 'Point Plot'}, {'type': 'barplot', 'title': 'Bar Plot (Mean)'},
            # Multivariate (uses all numeric columns)
            {'type': 'heatmap', 'title': 'Correlation Heatmap'}, {'type': 'pairplot', 'title': 'Pair Plot Grid (Sampled)'}
        ]
    elif column_type == 'categorical':
        charts_to_generate = [ {'type': 'countplot', 'title': f'Count Plot'} ]

    generated_charts = {}
    
    # --- Find helper columns for bivariate/categorical plots ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Find the first available numeric/categorical column that is NOT the target column
    other_numeric_col = next((col for col in numeric_cols if col != column_name), None)
    other_categorical_col = next((col for col in categorical_cols), None)


    for chart_info in charts_to_generate:
        fig = None 
        try:
            sns.set_theme(style="whitegrid", rc={
                "axes.facecolor": "#2C2541", "figure.facecolor": "#161220", "grid.color": "#4a3f6e", 
                "axes.labelcolor": "#a095c0", "xtick.color": "#a095c0", "ytick.color": "#a095c0", "text.color": "#E0D9F0"
            })
            
            chart_type = chart_info['type']
            
            # --- Charting Logic ---
            # Figure-level plots that need special handling
            if chart_type in ['jointplot', 'pairplot']:
                if chart_type == 'jointplot' and other_numeric_col:
                    g = sns.jointplot(data=df, x=column_name, y=other_numeric_col, kind="scatter", color="#9D4EDD")
                    g.fig.suptitle(f'Joint Plot: {column_name} vs {other_numeric_col}')
                    fig = g.fig
                elif chart_type == 'pairplot' and len(numeric_cols) > 1:
                    # IMPORTANT: Subsample for performance
                    df_sample = df.sample(n=min(len(df), 500))
                    g = sns.pairplot(df_sample[numeric_cols], corner=True)
                    g.fig.suptitle('Pair Plot of Numeric Columns (Sampled)')
                    fig = g.fig
            
            # Axes-level plots
            else:
                fig, ax = plt.subplots(figsize=(5, 4))
                
                # Univariate
                if chart_type == 'violin': sns.violinplot(data=df, y=column_name, ax=ax, color="#9D4EDD", cut=0)
                elif chart_type == 'boxplot': sns.boxplot(data=df, y=column_name, ax=ax, color="#9D4EDD")
                elif chart_type == 'displot': sns.histplot(data=df, x=column_name, kde=True, ax=ax, color="#9D4EDD")
                elif chart_type == 'kdeplot': sns.kdeplot(data=df, x=column_name, fill=True, ax=ax, color="#9D4EDD")
                elif chart_type == 'ecdfplot': sns.ecdfplot(data=df, x=column_name, ax=ax, color="#9D4EDD")
                elif chart_type == 'rugplot': sns.rugplot(data=df, x=column_name, ax=ax, color="#9D4EDD"); sns.kdeplot(data=df, x=column_name, ax=ax, color="#4a3f6e")

                # Bivariate
                elif other_numeric_col:
                    if chart_type == 'scatterplot': sns.scatterplot(data=df, x=column_name, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'hexbin': ax.hexbin(df[column_name], df[other_numeric_col], gridsize=20, cmap='viridis')
                    elif chart_type == 'kde2d': sns.kdeplot(data=df, x=column_name, y=other_numeric_col, fill=True, ax=ax)
                    elif chart_type == 'lineplot': sns.lineplot(data=df, x=column_name, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'regplot': sns.regplot(data=df, x=column_name, y=other_numeric_col, ax=ax, color="#9D4EDD")
                
                # Categorical vs Numeric
                elif other_categorical_col:
                    if chart_type == 'stripplot': sns.stripplot(data=df, x=column_name, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'swarmplot': sns.swarmplot(data=df, x=column_name, y=other_categorical_col, ax=ax, color="#9D4EDD", size=3)
                    elif chart_type == 'pointplot': sns.pointplot(data=df, x=column_name, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'barplot': sns.barplot(data=df, x=column_name, y=other_categorical_col, ax=ax, color="#9D4EDD")

                # Multivariate
                elif chart_type == 'heatmap':
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 6})

                # Categorical Univariate
                elif chart_type == 'countplot':
                    order = df[column_name].value_counts().index[:15]
                    sns.countplot(data=df, y=column_name, ax=ax, color="#9D4EDD", order=order)
                
                ax.set_title(chart_info['title'], color='white', fontsize=10)

            # --- Convert plot to Base64 ---
            if fig:
                plt.tight_layout()
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                base64_string = base64.b64encode(img_buffer.read()).decode('utf-8')
                generated_charts[chart_info['type']] = f"data:image/png;base64,{base64_string}"

        except Exception as e:
            print(f"Failed to generate chart '{chart_info.get('type')}': {e}")
        finally:
            if fig: plt.close(fig)
    
    return jsonify(generated_charts)

@current_app.route('/export_python_chart')
@login_required
def export_python_chart():
    """
    Generates a single chart and serves it as a file download (SVG or PNG).
    Uses query parameters for configuration.
    """
    # 1. Get parameters from the URL query string
    chart_type = request.args.get('chart_type')
    x_col = request.args.get('x_col')
    theme = request.args.get('theme', 'native')
    export_format = request.args.get('export_format', 'png')

    if not chart_type or not x_col:
        return "Missing parameters", 400

    fig = None 
    try:
        # 2. Load data and find helper columns
        df = utils.get_current_state_df()
        if x_col not in df.columns:
            return "Column not found", 404
            
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        other_numeric_col = next((col for col in numeric_cols if col != x_col), None)
        other_categorical_col = next((col for col in categorical_cols if col != x_col), None)

        # 3. Define and apply the selected theme
        native_theme_rc = { "axes.facecolor": "#2C2541", "figure.facecolor": "#161220", "grid.color": "#4a3f6e", "axes.labelcolor": "#a095c0", "xtick.color": "#a095c0", "ytick.color": "#a095c0", "text.color": "#E0D9F0", "patch.edgecolor": "#161220" }
        white_theme_rc = { "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF", "grid.color": "#EAEAEA", "axes.labelcolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#000000", "patch.edgecolor": "#FFFFFF" }

        if theme == 'white':
            sns.set_theme(style="whitegrid", rc=white_theme_rc)
            title_color = 'black'
        else:
            sns.set_theme(style="whitegrid", rc=native_theme_rc)
            title_color = 'white'

        # 4. Generate the requested plot (this logic is a mirror of the preview function)
        if chart_type in ['jointplot', 'pairplot']:
            if chart_type == 'jointplot' and other_numeric_col:
                g = sns.jointplot(data=df, x=x_col, y=other_numeric_col, kind="scatter", color="#9D4EDD")
                fig = g.fig
            elif chart_type == 'pairplot' and len(numeric_cols) > 1:
                df_sample = df.sample(n=min(len(df), 500))
                g = sns.pairplot(df_sample[numeric_cols], corner=True)
                fig = g.fig
        
        else:
            fig, ax = plt.subplots(figsize=(8, 6)) # Use a slightly larger figure for better export quality
            
            # NUMERIC PLOTS
            if chart_type == 'violin': sns.violinplot(data=df, y=x_col, ax=ax, color="#9D4EDD", cut=0)
            elif chart_type == 'boxplot': sns.boxplot(data=df, y=x_col, ax=ax, color="#9D4EDD")
            elif chart_type == 'displot': sns.histplot(data=df, x=x_col, kde=True, ax=ax, color="#9D4EDD")
            elif chart_type == 'kdeplot': sns.kdeplot(data=df, x=x_col, fill=True, ax=ax, color="#9D4EDD")
            elif chart_type == 'ecdfplot': sns.ecdfplot(data=df, x=x_col, ax=ax, color="#9D4EDD")
            elif chart_type == 'rugplot': sns.rugplot(data=df, x=x_col, ax=ax, color="#9D4EDD"); sns.kdeplot(data=df, x=x_col, ax=ax, color="#4a3f6e")
            elif chart_type in ['scatterplot', 'hexbin', 'kde2d', 'lineplot', 'regplot']:
                if other_numeric_col:
                    df_clean = df[[x_col, other_numeric_col]].dropna()
                    if chart_type == 'scatterplot': sns.scatterplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'hexbin': ax.hexbin(df_clean[x_col], df_clean[other_numeric_col], gridsize=20, cmap='viridis')
                    elif chart_type == 'kde2d': sns.kdeplot(data=df_clean, x=x_col, y=other_numeric_col, fill=True, ax=ax)
                    elif chart_type == 'lineplot': sns.lineplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'regplot': sns.regplot(data=df_clean, x=x_col, y=other_numeric_col, ax=ax, color="#9D4EDD")
                else: ax.text(0.5, 0.5, 'Requires a second numeric column', ha='center', va='center', color=title_color)
            elif chart_type in ['stripplot', 'swarmplot', 'pointplot', 'barplot'] and df[x_col].dtype in [np.number]:
                if other_categorical_col:
                    df_sample = df.sample(n=min(len(df), 1000))
                    if chart_type == 'stripplot': sns.stripplot(data=df_sample, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'swarmplot': sns.swarmplot(data=df_sample, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD", size=3)
                    elif chart_type == 'pointplot': sns.pointplot(data=df, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                    elif chart_type == 'barplot': sns.barplot(data=df, x=x_col, y=other_categorical_col, ax=ax, color="#9D4EDD")
                else: ax.text(0.5, 0.5, 'Requires a categorical column', ha='center', va='center', color=title_color)
            elif chart_type == 'heatmap' and df[x_col].dtype in [np.number]:
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 8})
                else: ax.text(0.5, 0.5, 'Requires multiple numeric columns', ha='center', va='center', color=title_color)

            # CATEGORICAL PLOTS
            elif chart_type == 'countplot':
                order = df[x_col].value_counts().index[:15]
                sns.countplot(data=df, y=x_col, ax=ax, color="#9D4EDD", order=order)
            elif chart_type == 'pieplot':
                counts = df[x_col].value_counts()
                top_n = counts.nlargest(7)
                if len(counts) > 7: top_n['Other'] = counts.iloc[7:].sum()
                ax.pie(top_n, labels=top_n.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(top_n)), textprops={'color': title_color})
            elif chart_type in ['cat_boxplot', 'cat_violinplot', 'cat_barplot']:
                if other_numeric_col:
                    order = df[x_col].value_counts().index[:15]
                    if chart_type == 'cat_boxplot': sns.boxplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order)
                    elif chart_type == 'cat_violinplot': sns.violinplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order, cut=0)
                    elif chart_type == 'cat_barplot': sns.barplot(data=df, x=other_numeric_col, y=x_col, ax=ax, color="#9D4EDD", order=order)
                else: ax.text(0.5, 0.5, 'Requires a numeric column', ha='center', va='center', color=title_color)
            elif chart_type in ['grouped_countplot', 'cat_heatmap']:
                if other_categorical_col:
                    if chart_type == 'grouped_countplot':
                        top_x_cats = df[x_col].value_counts().index[:10]
                        top_hue_cats = df[other_categorical_col].value_counts().index[:5]
                        df_filtered = df[df[x_col].isin(top_x_cats) & df[other_categorical_col].isin(top_hue_cats)]
                        sns.countplot(data=df_filtered, y=x_col, hue=other_categorical_col, ax=ax, order=top_x_cats)
                    elif chart_type == 'cat_heatmap':
                        contingency_table = pd.crosstab(df[x_col], df[other_categorical_col])
                        top_cols = contingency_table.sum().sort_values(ascending=False).index[:8]
                        contingency_table_sliced = contingency_table[top_cols]
                        show_annotations = contingency_table_sliced.shape[0] <= 10 and contingency_table_sliced.shape[1] <= 10
                        sns.heatmap(contingency_table_sliced, ax=ax, annot=show_annotations, fmt='d', cmap='viridis', annot_kws={"size": 8})
                        ax.tick_params(axis='x', rotation=90)
                else: ax.text(0.5, 0.5, 'Requires a second categorical column', ha='center', va='center', color=title_color)
            else:
                 ax.text(0.5, 0.5, 'Plot not applicable or data unavailable', ha='center', va='center', color=title_color)
            
            ax.set_title(chart_type.replace('_', ' ').replace('cat ', '').title(), color=title_color, fontsize=14)

        # 5. Configure for Export and Return Response
        if fig:
            if export_format == 'svg':
                mimetype = 'image/svg+xml'
                file_extension = 'svg'
            else:
                mimetype = 'image/png'
                file_extension = 'png'
            
            safe_col_name = "".join([c for c in x_col if c.isalnum()]).rstrip()
            filename = f"{chart_type}_of_{safe_col_name}.{file_extension}"

            img_buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(img_buffer, format=export_format, bbox_inches='tight')
            img_buffer.seek(0)
            
            return Response(
                img_buffer.getvalue(),
                mimetype=mimetype,
                headers={"Content-Disposition": f"attachment;filename={filename}"}
            )
        else:
            return "Could not generate chart figure.", 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Failed to generate chart", 500
    
    finally:
        # 6. CRITICAL: Close the figure to free up server memory
        if fig:
            plt.close(fig)


























@current_app.route('/convert_column_type', methods=['POST'])
def convert_column_type():
    """
    Converts a column to a specified data type using the state management system.
    """
    data = request.get_json()
    column_name = data.get('column_name')
    target_type = data.get('target_type')

    if not all([column_name, target_type]):
        return jsonify({"error": "Column name and target type are required."}), 400

    try:
        # 1. READ current state
        df = utils.get_current_state_df()
        
        if column_name not in df.columns:
            return jsonify({"error": f"Column '{column_name}' not found."}), 404

        # 2. PERFORM the conversion using Pandas
        # 'errors="coerce"' turns unparseable values into NaNs instead of crashing
        if target_type == 'numeric':
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        elif target_type == 'string':
            df[column_name] = df[column_name].astype(str)
            # Convert string 'nan' back to actual Nulls
            df[column_name] = df[column_name].replace('nan', None)
        elif target_type == 'datetime':
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        elif target_type == 'boolean':
            # A robust way to convert to boolean, handling strings like 'true'/'false'
            df[column_name] = df[column_name].map(
                lambda x: True if str(x).lower() in ['true', '1', 'yes'] 
                else (False if str(x).lower() in ['false', '0', 'no'] else None)
            ).astype('boolean') # Use nullable boolean type
        elif target_type == 'categorical':
            df[column_name] = df[column_name].astype('category')
        else:
            return jsonify({"error": f"Unknown target type: {target_type}"}), 400

        # 3. SAVE new state
        utils.save_new_state(df, description=f"Converted column '{column_name}' to '{target_type}'")
        
        return jsonify({
            "message": f"Successfully converted '{column_name}' to {target_type}.",
        })

    except Exception as e:
        current_app.logger.error(f"Error in '{request.endpoint}': {e}", exc_info=True)
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500

@current_app.route('/rename_column', methods=['POST'])
def rename_column():
    """
    Renames a specific column in the dataset using the state management system.
    """
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')

    if not all([old_name, new_name]):
        return jsonify({"error": "Old and new names must be provided."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()
        
        if old_name not in df_pd.columns:
            return jsonify({"error": f"Column '{old_name}' not found."}), 404
        if new_name in df_pd.columns and new_name != old_name:
            return jsonify({"error": f"Column name '{new_name}' already exists."}), 400

        # 2. PERFORM the operation (using Pandas in this case)
        df_renamed = df_pd.rename(columns={old_name: new_name})
        
        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_renamed, description=f"Renamed '{old_name}' to '{new_name}'")

        return jsonify({
            "message": f"Successfully renamed '{old_name}' to '{new_name}'.",
            "old_name": old_name,
            "new_name": new_name
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Rename failed: {str(e)}"}), 500
    
@current_app.route('/merge_columns', methods=['POST'])
def merge_columns():
    """
    Merges multiple columns into a single new column using the state management system.
    """
    data = request.get_json()
    columns_to_merge = data.get('columns_to_merge')
    new_column_name = data.get('new_column_name')
    separator = data.get('separator', '')
    drop_originals = data.get('drop_originals', False)

    if not columns_to_merge or len(columns_to_merge) < 2:
        return jsonify({"error": "Please select at least two columns to merge."}), 400
    if not new_column_name or not new_column_name.strip():
        return jsonify({"error": "New column name cannot be empty."}), 400

    try:
        # 1. READ state and CONVERT to Polars
        df_pd = utils.get_current_state_df()
        df_pl = pl.from_pandas(df_pd)
        
        # Validate that all selected columns exist
        for col in columns_to_merge:
            if col not in df_pl.columns:
                return jsonify({"error": f"Column '{col}' not found."}), 404
        
        if new_column_name in df_pl.columns:
            return jsonify({"error": f"Column name '{new_column_name}' already exists."}), 400

        # 2. PERFORM the Merge Operation using Polars
        merge_expression = pl.concat_str(
            [pl.col(c).cast(pl.Utf8) for c in columns_to_merge],
            separator=separator
        ).alias(new_column_name)
        
        df_merged_pl = df_pl.with_columns(merge_expression)
        
        if drop_originals:
            df_merged_pl = df_merged_pl.drop(columns_to_merge)
        
        # 3. CONVERT back to Pandas and SAVE the new state
        df_final_pd = df_merged_pl.to_pandas()
        utils.save_new_state(df_final_pd, description=f"Merged columns '{', '.join(columns_to_merge)}' into '{new_column_name}'")
        
        # Update session info
        session['col_count'] = len(df_final_pd.columns)
            
        return jsonify({
            "message": f"Successfully created column '{new_column_name}'.",
            "new_column": new_column_name
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Merge failed: {str(e)}"}), 500
    
@current_app.route('/split_column', methods=['POST'])
def split_column():
    """
    Splits a single text column into multiple new columns using the state management system.
    """
    data = request.get_json()
    column_to_split = data.get('column_to_split')
    delimiter = data.get('delimiter')
    new_column_names_str = data.get('new_column_names')
    drop_original = data.get('drop_original', False)

    if not all([column_to_split, delimiter, new_column_names_str]):
        return jsonify({"error": "Missing column, delimiter, or new column names."}), 400

    new_column_names = [name.strip() for name in new_column_names_str.split(',') if name.strip()]
    if not new_column_names:
        return jsonify({"error": "Please provide at least one new column name."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()
        
        if column_to_split not in df_pd.columns:
            return jsonify({"error": f"Column '{column_to_split}' not found."}), 404
        
        for new_name in new_column_names:
            if new_name in df_pd.columns:
                return jsonify({"error": f"New column name '{new_name}' already exists."}), 400

        # 2. PERFORM the Split Operation using Pandas
        # Ensure the target column is treated as a string to use .str accessor
        df_split = df_pd[column_to_split].astype(str).str.split(delimiter, expand=True)

        if len(new_column_names) != df_split.shape[1]:
            return jsonify({
                "error": f"The split resulted in {df_split.shape[1]} columns, but you provided {len(new_column_names)} names. Please provide a matching number of names."
            }), 400
        
        df_split.columns = new_column_names
        df_final = pd.concat([df_pd, df_split], axis=1)
        
        if drop_original:
            df_final.drop(columns=[column_to_split], inplace=True)
        
        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_final, description=f"Split column '{column_to_split}' into '{', '.join(new_column_names)}'")
        
        # Update session info
        session['col_count'] = len(df_final.columns)
            
        return jsonify({
            "message": f"Successfully split '{column_to_split}' into {len(new_column_names)} new columns.",
            "new_columns": new_column_names
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Split failed: {str(e)}"}), 500
    
@current_app.route('/drop_columns', methods=['POST'])
def drop_columns():
    """
    Removes one or more specified columns from the dataset using the state management system.
    """
    data = request.get_json()
    columns_to_drop = data.get('columns_to_drop')

    if not columns_to_drop or not isinstance(columns_to_drop, list) or len(columns_to_drop) == 0:
        return jsonify({"error": "Please select at least one column to drop."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()
        
        existing_columns = df_pd.columns.tolist()
        for col in columns_to_drop:
            if col not in existing_columns:
                return jsonify({"error": f"Column '{col}' not found in the dataset."}), 404

        # 2. PERFORM the Drop Operation using Pandas
        df_dropped = df_pd.drop(columns=columns_to_drop)
        
        if df_dropped.shape[1] == 0:
            return jsonify({"error": "Cannot drop all columns. The dataset must have at least one column remaining."}), 400

        # 3. SAVE the modified DataFrame as a new state
        utils.save_new_state(df_dropped, description=f"Dropped columns '{', '.join(columns_to_drop)}'")
        
        # Update session info
        session['col_count'] = len(df_dropped.columns)
            
        return jsonify({
            "message": f"Successfully dropped {len(columns_to_drop)} column(s).",
            "columns_dropped": columns_to_drop
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Drop failed: {str(e)}"}), 500
    







































@current_app.route('/formulate')
@login_required
def formulate_view():
    """Renders the Formulate (new column creation) view."""
    filename = session.get('filename', 'No file uploaded')
    return render_template('formulate.html', filename=filename)

@current_app.route('/apply_single_col_formula', methods=['POST'])
def apply_single_col_formula():
    """
    Applies a formula that uses one input column and an optional parameter.
    """
    data = request.get_json()
    formula_id = data.get('formula_id')
    input_col = data.get('input_col')
    new_col_name = data.get('new_col_name')
    param = data.get('parameter')

    if not all([formula_id, input_col, new_col_name]):
        return jsonify({"error": "Missing required fields."}), 400

    try:
        df_pd = utils.get_current_state_df()
        df_pl = pl.from_pandas(df_pd)

        if new_col_name in df_pl.columns:
            return jsonify({"error": f"Column name '{new_col_name}' already exists."}), 400

        col_expr = pl.col(input_col)
        expression = None

        # --- Map formula_id to the correct Polars expression ---
        if formula_id == 'add_constant': expression = col_expr + param
        elif formula_id == 'subtract_constant': expression = col_expr - param
        elif formula_id == 'multiply_by': expression = col_expr * param
        elif formula_id == 'divide_by': expression = col_expr / param
        elif formula_id == 'power': expression = col_expr.pow(param)
        elif formula_id == 'round': expression = col_expr.round(int(param))
        elif formula_id == 'floor': expression = col_expr.floor()
        elif formula_id == 'ceil': expression = col_expr.ceil()
        elif formula_id == 'abs': expression = col_expr.abs()
        elif formula_id == 'log': expression = (col_expr + 1).log()
        elif formula_id == 'log10': expression = (col_expr + 1).log10()
        elif formula_id == 'exp': expression = col_expr.exp()
        elif formula_id == 'sin': expression = col_expr.sin()
        elif formula_id == 'cos': expression = col_expr.cos()
        elif formula_id == 'tan': expression = col_expr.tan()
        else:
            return jsonify({"error": f"Unknown formula_id: {formula_id}"}), 400

        df_new = df_pl.with_columns(expression.alias(new_col_name))
        
        utils.save_new_state(df_new.to_pandas(), description=f"Applied formula '{formula_id}' to '{input_col}'")
        session['col_count'] = df_new.width

        return jsonify({"message": f"Successfully created column '{new_col_name}'."})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Formula application failed: {str(e)}"}), 500


@current_app.route('/apply_multi_col_formula', methods=['POST'])
def apply_multi_col_formula():
    """
    Applies a formula that uses two or more input columns.
    """
    data = request.get_json()
    formula_id = data.get('formula_id')
    input_cols = data.get('input_cols')
    new_col_name = data.get('new_col_name')

    if not all([formula_id, input_cols, new_col_name]) or len(input_cols) < 2:
        return jsonify({"error": "Missing fields or not enough columns selected."}), 400

    try:
        df_pd = utils.get_current_state_df()
        df_pl = pl.from_pandas(df_pd)
        
        if new_col_name in df_pl.columns:
            return jsonify({"error": f"Column name '{new_col_name}' already exists."}), 400

        expression = None
        
        if formula_id == 'add_cols': expression = pl.col(input_cols[0]) + pl.col(input_cols[1])
        elif formula_id == 'subtract_cols': expression = pl.col(input_cols[0]) - pl.col(input_cols[1])
        elif formula_id == 'multiply_cols': expression = pl.col(input_cols[0]) * pl.col(input_cols[1])
        elif formula_id == 'divide_cols': expression = pl.col(input_cols[0]) / pl.col(input_cols[1])
        elif formula_id == 'row_sum': expression = pl.sum_horizontal(input_cols)
        elif formula_id == 'row_mean': expression = pl.mean_horizontal(input_cols)
        elif formula_id == 'row_max': expression = pl.max_horizontal(input_cols)
        elif formula_id == 'row_min': expression = pl.min_horizontal(input_cols)
        else:
            return jsonify({"error": f"Unknown formula_id: {formula_id}"}), 400

        df_new = df_pl.with_columns(expression.alias(new_col_name))
        
        utils.save_new_state(df_new.to_pandas(), description=f"Applied formula '{formula_id}' to '{', '.join(input_cols)}'")
        session['col_count'] = df_new.width

        return jsonify({"message": f"Successfully created column '{new_col_name}'."})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Formula application failed: {str(e)}"}), 500
    
@current_app.route('/apply_single_col_text_formula', methods=['POST'])
def apply_single_col_text_formula():
    """
    Applies a formula that uses one input text column and optional parameters.
    """
    data = request.get_json()
    formula_id = data.get('formula_id')
    input_col = data.get('input_col')
    new_col_name = data.get('new_col_name')
    params = data.get('parameters', [])

    if not all([formula_id, input_col, new_col_name]):
        return jsonify({"error": "Missing required fields."}), 400

    try:
        df_pd = utils.get_current_state_df()
        df_pl = pl.from_pandas(df_pd)
        
        if new_col_name in df_pl.columns:
            return jsonify({"error": f"Column name '{new_col_name}' already exists."}), 400

        col_expr = pl.col(input_col).cast(pl.Utf8)
        expression = None

        if formula_id == 'to_upper': expression = col_expr.str.to_uppercase()
        elif formula_id == 'to_lower': expression = col_expr.str.to_lowercase()
        elif formula_id == 'to_title': expression = col_expr.str.to_titlecase()
        elif formula_id == 'trim_whitespace': expression = col_expr.str.strip_chars()
        elif formula_id == 'len_chars': expression = col_expr.str.len_chars()
        elif formula_id == 'slice': expression = col_expr.str.slice(params[0], params[1])
        elif formula_id == 'replace_literal': expression = col_expr.str.replace(params[0], params[1])
        elif formula_id == 'replace_regex': expression = col_expr.str.replace_all(params[0], params[1])
        elif formula_id == 'extract_regex': expression = col_expr.str.extract(params[0], 1)
        elif formula_id == 'split_get': expression = col_expr.str.split(params[0]).list.get(params[1])
        else:
            return jsonify({"error": f"Unknown formula_id: {formula_id}"}), 400

        df_new = df_pl.with_columns(expression.alias(new_col_name))
        
        utils.save_new_state(df_new.to_pandas(), description=f"Applied formula '{formula_id}' to '{input_col}'")
        session['col_count'] = df_new.width

        return jsonify({"message": f"Successfully created column '{new_col_name}'."})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Formula application failed: {str(e)}"}), 500


@current_app.route('/apply_multi_col_text_formula', methods=['POST'])
def apply_multi_col_text_formula():
    """
    Applies a text formula that uses two or more input columns (e.g., concatenate).
    """
    data = request.get_json()
    formula_id = data.get('formula_id')
    input_cols = data.get('input_cols')
    new_col_name = data.get('new_col_name')
    params = data.get('parameters', [])

    if not all([formula_id, input_cols, new_col_name]) or len(input_cols) < 2:
        return jsonify({"error": "Missing fields or not enough columns selected."}), 400

    try:
        df_pd = utils.get_current_state_df()
        df_pl = pl.from_pandas(df_pd)
        
        if new_col_name in df_pl.columns:
            return jsonify({"error": f"Column name '{new_col_name}' already exists."}), 400

        expression = None
        
        if formula_id == 'concatenate':
            separator = params[0] if params else ''
            exprs_to_concat = [pl.col(c).cast(pl.Utf8) for c in input_cols]
            expression = pl.concat_str(exprs_to_concat, separator=separator)
        else:
            return jsonify({"error": f"Unknown formula_id: {formula_id}"}), 400

        df_new = df_pl.with_columns(expression.alias(new_col_name))
        
        utils.save_new_state(df_new.to_pandas())
        utils.save_new_state(df_new.to_pandas(), description=f"Applied formula '{formula_id}' to '{', '.join(input_cols)}'")
        session['col_count'] = df_new.width

        return jsonify({"message": f"Successfully created column '{new_col_name}'."})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Formula application failed: {str(e)}"}), 500


























@current_app.route('/expression')
@login_required
def expression_view():
    """Renders the Expression (custom formula) view."""
    filename = session.get('filename', 'No file uploaded')
    return render_template('expression.html', filename=filename)

@current_app.route('/apply_expression', methods=['POST'])
def apply_expression():
    """
    Applies a complex, user-defined SQL-like expression to create a new column,
    using the state management system.
    """
    data = request.get_json()
    expression_str = data.get('expression')
    new_col_name = data.get('new_column_name')

    if not all([expression_str, new_col_name]):
        return jsonify({"error": "Missing expression or new column name."}), 400
    
    if not new_col_name.strip():
        return jsonify({"error": "New column name cannot be empty."}), 400

    try:
        # 1. READ the current state into a Pandas DataFrame
        df_pd = utils.get_current_state_df()

        # 2. CONVERT to a Polars DataFrame to use sql_expr
        # Use lazy evaluation for efficiency
        df_pl_lazy = pl.from_pandas(df_pd).lazy()

        if new_col_name in df_pl_lazy.columns:
            return jsonify({"error": f"Column name '{new_col_name}' already exists."}), 400

        # 3. PERFORM the operation using Polars' sql_expr
        df_new_lazy = df_pl_lazy.with_columns(
            sql_expr(expression_str).alias(new_col_name)
        )
        
        # Execute the lazy plan
        df_new_pl = df_new_lazy.collect()

        # 4. CONVERT the result back to a Pandas DataFrame
        modified_df = df_new_pl.to_pandas()
        
        # 5. SAVE the modified DataFrame as a new state
        utils.save_new_state(modified_df, description=f"Applied expression '{expression_str}' to '{new_col_name}'")

        # Update session info (col_count is now handled by utils.save_new_state implicitly via row_count)
        session['col_count'] = len(modified_df.columns)
        
        # Log the successful formula creation
        utils.log_formula_to_history(new_col_name, expression_str)

        return jsonify({
            "message": f"Successfully created column '{new_col_name}'.",
            "new_column": new_col_name
        })

    except Exception as e:
        # Polars often gives informative errors that we can pass back
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Expression Error: {str(e)}"}), 500
    
@current_app.route('/generate_expression', methods=['POST'])
@login_required
def generate_expression():
    """
    Uses AI to convert a natural language prompt into a Polars/SQL expression string.
    This version receives schema and function context directly from the front-end.
    """
    # 1. Configure AI Model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable not set."}), 500
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite') # Using the lighter model is fine here
    except Exception as e:
        return jsonify({"error": f"Failed to configure AI model: {e}"}), 500

    # 2. Get data from the request body (same as SQL generator)
    request_data = request.get_json()
    user_prompt = request_data.get('prompt')
    dataset_context = request_data.get('context') # The frontend will now provide this

    if not user_prompt or not dataset_context:
        return jsonify({"error": "Missing prompt or dataset context."}), 400

    try:
        # 3. Assemble the prompt (updated to focus on expressions)
        prompt = f"""
You are an expert AI that translates natural language into a single, valid SQL/Polars expression string for creating a new column.

**CRITICAL RULES:**
1.  Your response MUST be ONLY the expression string itself, with no explanations or markdown.
2.  If you cannot create a valid expression, you MUST return the single character `0`.
3.  You MUST use column names exactly as they appear in the schema, enclosing them in double quotes (e.g., "SalePrice").

---
**CONTEXT & AVAILABLE TOOLS:**
{dataset_context}
---

**USER REQUEST:** "{user_prompt}"

Based on all the rules and context, generate the expression string now.
"""

        # 4. Call the AI and return the result
        response = model.generate_content(prompt)
        expression = response.text.strip()
        
        return jsonify({"expression": expression})

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during AI processing: {str(e)}"}), 500

@current_app.route('/get_formula_history')
def get_formula_history():
    """API endpoint to fetch the contents of the formula history log file."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'formula_history.json')
        
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                logs = json.load(f)
                return jsonify(logs)
        else:
            return jsonify([])
    except Exception as e:
        print(f"Error reading formula history log: {e}")
        return jsonify({"error": str(e)}), 500
    


































# ===================================================================
# === NEW SECURE AI CHAT SYSTEM
# ===================================================================

@current_app.route('/ask_ai_secure', methods=['POST'])
@login_required
def ask_ai_secure():
    """
    Handles AI chat requests with a self-correction loop and conversational memory.
    """
    # 1. Configure AI Models
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable not set."}), 500
    try:
        genai.configure(api_key=api_key)
        sql_model = genai.GenerativeModel('gemini-2.5-flash')
        nlp_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    except Exception as e:
        return jsonify({"error": f"Failed to configure AI model: {e}"}), 500

    # 2. Get User Input
    request_data = request.get_json()
    user_question = request_data.get('question')
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # 3. Generate Secure Context & Load Conversational Memory
        dataset_context = utils.generate_secure_ai_context()
        if "Error:" in dataset_context:
            return jsonify({"error": dataset_context}), 500

        # --- NEW: Load and format the last 5 chat interactions ---
        conversation_history_str = ""
        try:
            org_id = session['org_id']
            org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
            log_filepath = os.path.join(org_folder, 'insights_chat_log.json')
            if os.path.exists(log_filepath):
                with open(log_filepath, 'r') as f:
                    logs = json.load(f)[-5:] # Get the last 5 interactions
                    if logs:
                        history_lines = ["### CONVERSATION HISTORY"]
                        for log in logs:
                            history_lines.append(f"User: {log['user']}")
                            history_lines.append(f"AI: {log['ai']}")
                        conversation_history_str = "\n".join(history_lines)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            # Continue without history if loading fails

        # --- NEW: Self-Correction Loop ---
        max_retries = 3
        error_history_str = ""
        generated_sql = ""
        query_result_df = None

        for attempt in range(max_retries):
            try:
                # 4. Generate SQL Query
                prompt_to_sql = f"""
You are an expert DuckDB data analyst. Your task is to convert a user's question into a single, valid SQL query.

**CRITICAL RULES:**
1.  Your response MUST be ONLY the SQL query itself. Do not include any explanations, markdown, or any text other than the SQL.
2.  The ONLY valid table name is "Dataset". You MUST enclose this table name and all column names in double quotes (e.g., SELECT "SalePrice" FROM "Dataset").
3.  Use the context, history, and previous errors below to answer the user's LATEST question.

{dataset_context}

{conversation_history_str}

{error_history_str}
---
**LATEST User Question:** "{user_question}"

Based on all the rules and context, generate the SQL query now.
"""
                sql_response = sql_model.generate_content(prompt_to_sql)
                generated_sql = sql_response.text.strip().replace('```sql', '').replace('```', '').strip()

                # 5. Execute SQL Query
                current_df = utils.get_current_state_df()
                con = duckdb.connect()
                con.register('Dataset', current_df)
                query_result_df = con.execute(generated_sql).fetchdf()

                print(f"Attempt {attempt + 1} successful!")
                break # Exit the loop on success

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                error_message = str(e)
                # Format the error history for the next prompt iteration
                error_history_str += (
                    f"\n--- PREVIOUS ATTEMPT FAILED ---\n"
                    f"The last SQL query you generated was:\n```sql\n{generated_sql}\n```\n"
                    f"It failed with the error: \"{error_message}\"\n"
                    f"Please analyze the error and the schema, and provide a corrected SQL query.\n"
                )
                if attempt == max_retries - 1:
                    # If all retries fail, raise the last exception
                    raise e
        
        # This check is necessary in case the loop completes without ever succeeding
        if query_result_df is None:
            raise Exception("The AI was unable to generate a valid SQL query after multiple attempts.")

        # 6. Generate Natural Language Answer
        prompt_to_nlp = f"""
You are a friendly data analyst assistant. A user asked a question, a SQL query was run, and here is the result as a JSON object.
Please provide a clear, natural language answer to the original question based ONLY on the data provided. Use markdown for emphasis.

**Original Question:** "{user_question}"
**SQL Query Result (JSON):**
{query_result_df.to_json(orient='records')}

---
**Your Answer:**
"""
        nlp_response = nlp_model.generate_content(prompt_to_nlp)
        natural_language_answer = nlp_response.text.strip()
        
        # 7. Log and Respond
        utils.log_chat_interaction(user_question, natural_language_answer)
        
        return jsonify({
            "natural_language_answer": natural_language_answer,
            "generated_sql": generated_sql
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        error_message = f"An error occurred during AI processing: {str(e)}"
        # Log the user question and the final error to the chat log for context
        utils.log_chat_interaction(user_question, f"Error: {error_message}")
        return jsonify({"error": error_message}), 500
    

@current_app.route('/get_chat_logs')
@login_required
def get_chat_logs():
    """API endpoint to fetch the contents of the chat log file for the user's org."""
    try:
        org_id = session['org_id']
        org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
        log_filepath = os.path.join(org_folder, 'insights_chat_log.json')
            
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                logs = json.load(f)
                return jsonify(logs)
        else:
            return jsonify([]) # No history yet, return empty list
    except Exception as e:
        print(f"Error reading chat log: {e}")
        return jsonify({"error": str(e)}), 500
    










































@current_app.route('/pipeline_creator')
def pipeline_creator():
    """
    Renders the pipeline creator, providing all necessary data (schema and
    a default pipeline spec) in the initial request.
    """
    filename = session.get('filename')
    if not filename:
        flash("Please upload a file first.", "error")
        return redirect(url_for('upload_file'))

    schema = utils._get_pipeline_creator_schema()
    default_pipeline = utils._generate_default_pipeline_for_creator(filename, schema)
    
    return render_template(
        'pipeline_creator.html', 
        pipeline_spec_json=json.dumps(default_pipeline),
        schema_json=json.dumps(schema)
    )


# ===================================================================
# === PIPELINE MANAGEMENT (DATABASE-BACKED)
# ===================================================================

@current_app.route('/pipelines/save', methods=['POST'])
@login_required
def save_pipeline():
    data = request.get_json()
    name = data.get('name')
    steps = data.get('steps')
    if not name or not steps:
        return jsonify({"error": "Missing pipeline name or steps."}), 400

    try:
        org_id = session['org_id']
        
        existing = g.supabase.table('pipelines').select('id').eq('name', name).eq('organization_id', org_id).execute()
        
        if existing.data:
            # If it exists, it's an update, so no metric change
            g.supabase.table('pipelines').update({'steps': steps}).eq('id', existing.data[0]['id']).execute()
            message = f"Pipeline '{name}' updated successfully."
        else:
            # If it doesn't exist, it's a new pipeline
            g.supabase.table('pipelines').insert({
                'name': name,
                'steps': steps,
                'organization_id': org_id
            }).execute()
            
            # Increment the pipeline_count metric
            update_metric(org_id, 'pipeline_count', 1)
            message = f"Pipeline '{name}' saved successfully."
            
        return jsonify({"message": message})
    except Exception as e:
        print(f"Error in save_pipeline: {e}")
        return jsonify({"error": f"Failed to save pipeline: {str(e)}"}), 500

@current_app.route('/pipelines/list', methods=['GET'])
@login_required
def list_pipelines():
    """Lists all pipelines for the current organization from the database."""
    try:
        org_id = session['org_id']
        
        # Use the pre-authenticated client from g
        response = g.supabase.table('pipelines').select('name').eq('organization_id', org_id).order('name').execute()
        
        pipelines = [item['name'] for item in response.data]
        
        return jsonify({"pipelines": pipelines})
    except Exception as e:
        print(f"Error in list_pipelines: {e}")
        return jsonify({"error": f"Failed to list pipelines: {str(e)}"}), 500

@current_app.route('/pipelines/load/<pipeline_name>', methods=['GET'])
@login_required
def load_pipeline(pipeline_name):
    """Loads a specific pipeline's details from the database."""
    try:
        org_id = session['org_id']

        # Use the pre-authenticated client from g
        response = g.supabase.table('pipelines').select('name, steps').eq('name', pipeline_name).eq('organization_id', org_id).single().execute()

        if not response.data:
            return jsonify({"error": "Pipeline not found."}), 404
        
        return jsonify(response.data)
    except Exception as e:
        print(f"Error in load_pipeline: {e}")
        return jsonify({"error": f"Failed to load pipeline: {str(e)}"}), 500

@current_app.route('/pipelines/delete/<pipeline_name>', methods=['DELETE'])
@login_required
def delete_pipeline(pipeline_name):
    try:
        org_id = session['org_id']

        # Delete the pipeline row from the database
        g.supabase.table('pipelines').delete().eq('name', pipeline_name).eq('organization_id', org_id).execute()
        
        # Decrement the pipeline_count metric
        update_metric(org_id, 'pipeline_count', -1)
        
        return jsonify({"message": f"Pipeline '{pipeline_name}' deleted."})
    except Exception as e:
        print(f"Error in delete_pipeline: {e}")
        return jsonify({"error": f"Failed to delete pipeline: {str(e)}"}), 500

@current_app.route('/pipelines/rename', methods=['POST'])
@login_required
def rename_pipeline():
    """Renames a specific pipeline in the database."""
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    if not old_name or not new_name:
        return jsonify({"error": "Old and new names must be provided."}), 400

    try:
        org_id = session['org_id']
        
        # Use the pre-authenticated client from g
        existing = g.supabase.table('pipelines').select('id').eq('name', new_name).eq('organization_id', org_id).execute()
        if existing.data:
            return jsonify({"error": f"A pipeline named '{new_name}' already exists."}), 409

        g.supabase.table('pipelines').update({'name': new_name}).eq('name', old_name).eq('organization_id', org_id).execute()
        
        return jsonify({"message": f"Pipeline renamed to '{new_name}'."})
    except Exception as e:
        print(f"Error in rename_pipeline: {e}")
        return jsonify({"error": f"Failed to rename pipeline: {str(e)}"}), 500
    
# ===================================================================
# === PIPELINE CODE GENERATION & EXECUTION
# ===================================================================

@current_app.route('/generate_pipeline_code', methods=['POST'])
def generate_pipeline_code():
    data = request.get_json()
    steps = data.get('steps', [])
    pipeline_name = data.get('name', 'unnamed_pipeline')
    
    # Sanitize pipeline name for use in a filename
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-]', '', pipeline_name).strip().replace(' ', '_')
    output_filename = f"{safe_filename}_cleaned.csv"

    # --- Start of script with necessary imports ---
    script_parts = [
        "import pandas as pd",
        "import numpy as np",
        "import os",
        "from sklearn.preprocessing import PowerTransformer",
        "import re",
        "\n",
        "def run_pipeline(data_path):"
    ]
    
    # --- Always add the load step ---
    script_parts.append("    # Step 1: Load Initial Dataset from Parquet file")
    script_parts.append("    df = pd.read_parquet(data_path)")
    script_parts.append("    print(f'Initial shape: {df.shape}')\n")

    # --- Process user-defined steps ---
    if not steps:
        script_parts.append("    # No processing steps were defined.")
    else:
        for step in steps:
            step_code = utils._generate_step_code(step)
            indented_code = "\n".join(["    " + line for line in step_code.splitlines()])
            script_parts.append(indented_code)

    # --- Always add the export step ---
    script_parts.append(f"\n    # Final Step: Export Cleaned Data")
    script_parts.append(f"    output_path = os.path.join(os.path.dirname(data_path), '{output_filename}')")
    script_parts.append(f"    df.to_csv(output_path, index=False)")
    script_parts.append(f"    print(f'Successfully exported cleaned data to {{output_path}}')\n")
    
    # --- Always return the dataframe for in-app use (like previewing) ---
    script_parts.append("    return df")

    full_script = "\n".join(script_parts)
    return jsonify({"code": full_script})


@current_app.route('/execute_pipeline_code', methods=['POST'])
def execute_pipeline_code():
    data = request.get_json()
    script_code = data.get('code')
    if not script_code:
        return jsonify({"error": "No script provided."}), 400
    try:
        history_dir = utils.get_history_dir()
        initial_state_path = os.path.join(history_dir, 'state_0.parquet')
        if not os.path.exists(initial_state_path):
            return jsonify({"error": "Initial dataset not found."}), 404
        
        # --- MODIFIED: Enqueue the job instead of running it ---
        job = current_app.task_queue.enqueue(
            execute_pipeline_code_task,
            args=(script_code, initial_state_path, history_dir),
            job_timeout='1h' # 1 hour timeout
        )
        
        return jsonify({"job_id": job.get_id()}), 202 # Return 202 Accepted
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"An error occurred in this route: {e}", exc_info=True)
        return jsonify({"error": f"Failed to start pipeline job: {str(e)}"}), 500


















































# ===================================================================
# === PIPELINE COMPARATOR & BENCHMARKING
# ===================================================================

@current_app.route('/pipeline_comparator')
@login_required
def pipeline_comparator_view():
    """Renders the Pipeline Comparator view."""
    return render_template('pipeline_comparator.html')

@current_app.route('/get_comparator_initial_data')
@login_required  # Add the decorator to secure the endpoint
def get_comparator_initial_data():
    """
    API to fetch saved pipelines FROM THE DATABASE and a dynamically determined schema.
    """
    try:
        # --- THE FIX IS HERE: Fetch pipelines from Supabase ---
        org_id = session['org_id']
        # Use the pre-authenticated client provided by the decorator
        response = g.supabase.table('pipelines').select('name').eq('organization_id', org_id).execute()
        
        if response.data:
            pipeline_names = [item['name'] for item in response.data]
        else:
            pipeline_names = []
        # --- END OF FIX ---

        # --- The schema detection logic below is correct and remains unchanged ---
        # It correctly reads the raw data file for the current organization.
        history_dir = utils.get_history_dir()
        raw_df = pd.read_parquet(os.path.join(history_dir, 'raw.parquet'))
        schema = []
        
        NUMERIC_CARDINALITY_THRESHOLD = 100 

        for col_name in raw_df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(raw_df[col_name])
            cardinality = raw_df[col_name].nunique()
            
            col_info = {'name': col_name}

            if is_numeric:
                if cardinality < NUMERIC_CARDINALITY_THRESHOLD:
                    col_info['type'] = 'categorical'
                else:
                    col_info['type'] = 'numeric'
            else:
                col_info['type'] = 'categorical'
            
            if col_info['type'] == 'categorical':
                col_info['cardinality'] = cardinality

            schema.append(col_info)

        return jsonify({
            "pipelines": sorted(pipeline_names),
            "schema": schema
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to load initial data: {str(e)}"}), 500

# --- NEW STAGE 1 ENDPOINT ---
@current_app.route('/run_pipelines_for_comparison', methods=['POST'])
def run_pipelines_for_comparison():
    data = request.get_json()
    pipeline_a_spec = data.get('pipeline_a_spec')
    pipeline_b_spec = data.get('pipeline_b_spec')

    use_raw_dataset = data.get('use_raw_dataset', True)

    if not all([pipeline_a_spec, pipeline_b_spec]):
        return jsonify({"error": "Missing pipeline specifications."}), 400

    try:
        history_dir = utils.get_history_dir()
        
        # --- MODIFIED: Enqueue the job ---
        job = current_app.task_queue.enqueue(
            run_pipelines_for_comparison_task,
            args=(history_dir, pipeline_a_spec, pipeline_b_spec, use_raw_dataset),
            job_timeout='1h'
        )
        
        return jsonify({"job_id": job.get_id()}), 202
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred while starting the job: {str(e)}"}), 500


# --- NEW STAGE 2 ENDPOINT ---
@current_app.route('/run_benchmark_for_comparison', methods=['POST'])
@login_required
def run_benchmark_for_comparison():
    """
    DISPATCHER ROUTE: Enqueues a background job to run the PyCaret benchmarks.
    This route receives the benchmark configuration, creates a job, and
    immediately returns a job ID to the client for polling.
    """
    data = request.get_json()
    target = data.get('target')
    problem_type = data.get('problem_type')
    pipeline_a_spec = data.get('pipeline_a_spec')
    pipeline_b_spec = data.get('pipeline_b_spec')
    time_a = data.get('execution_time_a')
    time_b = data.get('execution_time_b')

    # 1. Validate that all required parameters were sent from the frontend
    if not all([target, problem_type, pipeline_a_spec, pipeline_b_spec, time_a is not None, time_b is not None]):
        return jsonify({"error": "Missing benchmark parameters or pipeline specs."}), 400

    try:
        # 2. Get the path to the directory containing the processed data files.
        #    This path will be passed to the background worker.
        history_dir = utils.get_history_dir()

        # 3. Enqueue the job.
        #    The first argument is the function to run (from tasks.py).
        #    'args' is a tuple of arguments to pass to that function.
        #    'job_timeout' tells the worker to stop if the task runs longer than this.
        job = current_app.task_queue.enqueue(
            run_benchmark_for_comparison_task,
            args=(
                history_dir,
                target,
                problem_type,
                pipeline_a_spec,
                pipeline_b_spec,
                time_a,
                time_b
            ),
            job_timeout='3h'  # Set a generous 3-hour timeout for the benchmark
        )

        # 4. Immediately return a 202 Accepted response with the job ID.
        #    The frontend will use this ID to poll for the result.
        return jsonify({"job_id": job.get_id()}), 202

    except Exception as e:
        # This block catches errors during the job submission itself (e.g., Redis is down).
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during benchmark job submission: {str(e)}"}), 500
    
















































def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'admin':
            # Abort with 403 Forbidden if not an admin
            abort(403) 
        return f(*args, **kwargs)
    return decorated_function

@current_app.route('/admin', methods=['GET'])
@login_required
@admin_required
def admin_dashboard():
    """
    Handles the GET request for the main Admin Dashboard page. 
    It fetches all team members and usage metrics for the administrator's 
    organization and renders the admin template.
    """
    org_id = session['org_id']
    
    # Initialize variables to hold data for the template
    members = []
    usage = {}

    try:
        # --- Step 1: Fetch all profiles for the admin's organization ---
        # Use the ADMIN client to bypass RLS and securely get all profiles for THIS org
        profiles_res = supabase_admin.table('profiles').select('id, role').eq('organization_id', org_id).execute()
        
        # Proceed only if profile data is found
        if profiles_res.data:
            # Create a dictionary for quick lookups: {user_id: role}
            profiles_map = {p['id']: p['role'] for p in profiles_res.data}
            
            # --- Step 2: Get the master list of all users from Supabase Auth ---
            all_users_res = supabase_admin.auth.admin.list_users()
            # The response object structure can vary slightly, so check for 'users' attribute
            user_list = all_users_res.users if hasattr(all_users_res, 'users') else all_users_res

            # --- Step 3: Filter the master user list ---
            # Create the final 'members' list by including only users whose IDs are in our organization's profiles_map
            for user in user_list:
                if user.id in profiles_map:
                    # Append a dictionary containing the full user object and their specific role
                    members.append({'id': user, 'role': profiles_map[user.id]})
        
        # --- Step 4: Fetch usage metrics for the organization ---
        # Use the regular, user-level client for this call since it's allowed by RLS policies
        supabase = get_supabase() 
        metrics_res = supabase.table('usage_metrics').select('*').eq('organization_id', org_id).single().execute()
        if metrics_res.data:
            usage = metrics_res.data

        # --- Step 5: Render the template with the fetched data ---
        return render_template('admin.html', members=members, usage=usage)

    except Exception as e:
        # If any part of the data fetching fails, flash an error message
        flash(f"Error loading dashboard data: {e}", "error")
        # Render the page with empty data to prevent a crash
        return render_template('admin.html', members=[], usage={})


@current_app.route('/admin/invite', methods=['POST'])
@login_required
@admin_required
def invite_user():
    # 1. Get the source page from the form, defaulting to the main admin dashboard.
    source_page = request.form.get('source_page', 'admin_dashboard')
    
    email = request.form.get('email')
    password = request.form.get('password')
    org_id = session['org_id']
    new_user_id = None

    try:
        # The core user creation logic remains exactly the same
        user_res = supabase_admin.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
        if not user_res.user:
            raise Exception(f"Supabase Auth Error: Could not create user. They may already exist.")
        new_user_id = user_res.user.id

        profile_res = supabase_admin.table('profiles').insert({ "id": new_user_id, "organization_id": org_id, "role": "user" }).execute()
        if not profile_res.data:
            raise Exception("Database Error: Could not create user profile.")

        update_metric(org_id, 'user_count', 1)
        current_app.logger.warning(f"Admin invited new user '{email}'.")
        flash(f"Successfully invited user {email}.", "success")

    except Exception as e:
        if new_user_id:
            supabase_admin.auth.admin.delete_user(new_user_id) # Rollback
        flash(str(e), "error")
    
    # 2. Redirect back to the page the request came from.
    return redirect(url_for(source_page))

    
def update_metric(org_id, metric_name, value):
    """
    Safely increments a usage metric for an organization using a remote procedure call.
    :param org_id: The UUID of the organization.
    :param metric_name: The column name ('user_count', 'pipeline_count', 'rows_processed').
    :param value: The integer value to add (can be negative, e.g., -1 for deletion).
    """
    try:
        # Use the admin client to call the database function
        supabase_admin.rpc('increment_metric', {
            'org_id': str(org_id),
            'metric_name': metric_name,
            'increment_value': int(value)
        }).execute()
    except Exception as e:
        # Log this critical error. You need to know if metrics are failing.
        print(f"!!! CRITICAL: FAILED TO UPDATE METRIC '{metric_name}' for org '{org_id}'. Error: {e}")

@current_app.route('/admin/set_role/<user_id>', methods=['POST'])
@login_required
@admin_required
def set_role(user_id):
    # --- THIS IS THE FIX ---
    # 1. Get the source page from the form, defaulting to 'admin_dashboard' for safety.
    source_page = request.form.get('source_page', 'admin_dashboard')
    # --- END OF FIX ---
    
    new_role = request.form.get('role')
    if new_role not in ['admin', 'user']:
        flash("Invalid role specified.", "error")
        return redirect(url_for(source_page)) # Use the dynamic source page

    if user_id == session['user']['id']:
        flash("You cannot change your own role.", "error")
        return redirect(url_for(source_page)) # Use the dynamic source page

    try:
        profile_to_change = supabase_admin.table('profiles').select('organization_id').eq('id', user_id).single().execute()
        if not profile_to_change.data or profile_to_change.data['organization_id'] != session['org_id']:
            abort(403)

        supabase_admin.table('profiles').update({'role': new_role}).eq('id', user_id).execute()
        current_app.logger.warning(f"Admin set role for user '{user_id}' to '{new_role}'.")
        flash(f"User role has been updated to '{new_role}'.", "success")

    except Exception as e:
        flash(f"Error updating role: {e}", "error")

    # --- THIS IS THE FIX ---
    # 2. Redirect back to the page the request came from.
    return redirect(url_for(source_page))
    # --- END OF FIX ---

@current_app.route('/admin/remove_user/<user_id>', methods=['POST'])
@login_required
@admin_required
def remove_user(user_id):
    # --- THIS IS THE FIX ---
    # 1. Get the source page from the form.
    source_page = request.form.get('source_page', 'admin_dashboard')
    # --- END OF FIX ---

    if user_id == session['user']['id']:
        flash("You cannot remove yourself.", "error")
        return redirect(url_for(source_page)) # Use the dynamic source page

    try:
        profile_to_remove = supabase_admin.table('profiles').select('organization_id').eq('id', user_id).single().execute()
        if not profile_to_remove.data or profile_to_remove.data['organization_id'] != session['org_id']:
            abort(403) 

        supabase_admin.auth.admin.delete_user(user_id)
        update_metric(session['org_id'], 'user_count', -1)
        current_app.logger.warning(f"Admin removed user '{user_id}'.")
        flash("User removed successfully.", "success")

    except Exception as e:
        flash(f"Error removing user: {e}", "error")

    # --- THIS IS THE FIX ---
    # 2. Redirect back to the source page.
    return redirect(url_for(source_page))
    # --- END OF FIX ---

































# ===================================================================
# === NEW JOB STATUS ENDPOINT
# ===================================================================













@current_app.route('/commit_state_to_source', methods=['POST'])
@login_required
def commit_state_to_source():
    """Dispatches a job to overwrite the original source file with the current data state."""
    try:
        history_dir = utils.get_history_dir()
        source_filepath = session.get('filepath')
        if not source_filepath:
            return jsonify({"error": "Original file path not found in session."}), 400

        job = current_app.task_queue.enqueue(
            commit_state_to_source_task,
            args=(history_dir, source_filepath),
            job_timeout='10m'
        )
        return jsonify({"job_id": job.get_id()}), 202
    except Exception as e:
        return jsonify({"error": f"Failed to start save job: {str(e)}"}), 500
    
@current_app.route('/save_state_as_new_file', methods=['POST'])
@login_required
def save_state_as_new_file():
    """Validates a new filename and dispatches a job to save the current state to it."""
    data = request.get_json()
    new_filename_raw = data.get('new_filename')

    if not new_filename_raw:
        return jsonify({"error": "New filename not provided."}), 400

    new_filename = secure_filename(new_filename_raw)
    if not new_filename.lower().endswith('.csv'):
        new_filename += '.csv'

    try:
        # Use the correct helper to get the .../datasets/ directory
        org_datasets_folder = utils.get_org_datasets_dir()
        new_filepath = os.path.join(org_datasets_folder, new_filename)

        if os.path.exists(new_filepath):
            return jsonify({"error": f"A file named '{new_filename}' already exists."}), 409

        history_dir = utils.get_history_dir()
        job = current_app.task_queue.enqueue(
            save_state_as_new_file_task,
            args=(history_dir, new_filepath),
            job_timeout='10m'
        )
        return jsonify({"job_id": job.get_id()}), 202
    except Exception as e:
        current_app.logger.error(f"Error in 'save_state_as_new_file': {e}", exc_info=True)
        return jsonify({"error": f"Failed to start 'Save As' job: {str(e)}"}), 500

@current_app.route('/switch_session_to_file', methods=['POST'])
@login_required
def switch_session_to_file():
    """Updates the user's session to point to a new file after a 'Save As' operation."""
    data = request.get_json()
    new_filepath = data.get('new_filepath')

    if not new_filepath:
        return jsonify({"error": "Filepath not provided."}), 400

    # --- Security Check: Ensure the new path is within the user's organization folder ---
    org_id = session.get('org_id')
    org_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(org_id))
    if not os.path.abspath(new_filepath).startswith(os.path.abspath(org_folder)):
        abort(403) # Forbidden access

    try:
        # Update session variables to reflect the new working file
        df = pd.read_csv(new_filepath)
        session['filepath'] = new_filepath
        session['filename'] = os.path.basename(new_filepath)
        session['row_count'] = len(df)
        session['col_count'] = len(df.columns)
        session['file_size'] = utils.format_bytes(os.path.getsize(new_filepath))
        
        # This will create a *new* history for the "saved as" file
        utils.initialize_new_history_from_df(df)

        return jsonify({"success": True, "message": "Session updated."})
    except Exception as e:
        return jsonify({"error": f"Failed to switch session: {str(e)}"}), 500
    
















# ===================================================================
# === GLOBAL JOB STATUS ROUTES (for base.html)
# ===================================================================

@current_app.route('/job_progress/<job_id>')
@login_required
def job_progress(job_id):
    """
    Fetches the latest progress metadata for a given job from ANY configured queue.
    """
    job = None
    # --- FIX: Iterate through all configured queues to find the job ---
    # This ensures that jobs from 'low' or 'high' priority queues can be found.
    for queue_name in current_app.config['QUEUES']:
        q = Queue(queue_name, connection=current_app.redis)
        found_job = q.fetch_job(job_id)
        if found_job is not None:
            job = found_job
            break
    # --- END OF FIX ---

    if job is None:
        # This now correctly means the job is truly finished and has been cleaned up, or it never existed.
        return jsonify({
            'status': 'finished', 
            'progress': 100, 
            'status_message': 'Job complete or not found.'
        })
    
    # The rest of the function logic remains the same
    progress = job.meta.get('progress', 0)
    status_message = job.meta.get('status_message', 'Waiting in queue...')
    
    response_data = {
        'status': job.get_status(),
        'progress': progress,
        'status_message': status_message
    }
    
    # If the job is finished or failed, include the final result/error
    if job.is_finished:
        response_data['result'] = job.result
    elif job.is_failed:
        response_data['error'] = str(job.exc_info).strip()
    
    return jsonify(response_data)

@current_app.route('/get_current_job')
@login_required
def get_current_job():
    """
    Checks the session for an active job and returns its ID and name.
    """
    if 'active_job_id' in session and 'active_job_name' in session:
        return jsonify({
            "job_id": session['active_job_id'],
            "job_name": session['active_job_name']
        })
    else:
        # Return an empty object if no job is active
        return jsonify({})

@current_app.route('/clear_job_status', methods=['POST'])
@login_required
def clear_job_status():
    """
    Removes the active job information from the user's session.
    This is called by the frontend once a job is complete.
    """
    session.pop('active_job_id', None)
    session.pop('active_job_name', None)
    return jsonify({"success": True})