# ===================================================================
# === Stage 1: Build Stage ===
# ===================================================================
# Use a full Python image to build dependencies. This stage includes the
# necessary compilers and OS libraries to build packages from source if needed.
FROM python:3.11 as builder

# Set the working directory for this build stage
WORKDIR /app

# Install OS-level dependencies required by some Python packages.
# For example, libpq-dev is needed by psycopg2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment within the builder stage. This isolates dependencies.
RUN python -m venv /opt/venv
# Add the virtual environment's bin directory to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first to leverage Docker's build cache.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies into the virtual environment.
# --no-cache-dir is used to reduce the final image size.
RUN pip install --no-cache-dir -r requirements.txt


# ===================================================================
# === Stage 2: Final Production Stage ===
# ===================================================================
# Start from a slim Python base image for a smaller, more secure final container.
FROM python:3.11-slim

# Create the dedicated, unprivileged user and their home directory first.
# This ensures the home directory exists with the correct ownership before we use it.
RUN useradd --create-home appuser

# Install the OpenMP library (libgomp1) required by LightGBM and XGBoost,
# which are dependencies of PyCaret. This is missing in the 'slim' image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully populated virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Set the working directory to the user's home directory. All subsequent
# commands and the application itself will run from here.
WORKDIR /home/appuser

# Copy the application source code into the new working directory.
COPY . .

# Change the ownership of all application files to the non-root user.
# This is a critical security step and gives the app user full control
# over its own files, resolving the PermissionError.
RUN chown -R appuser:appuser /home/appuser

# Switch to the non-root user for all subsequent commands and for running the app.
USER appuser

# Activate the virtual environment by adding it to the PATH.
ENV PATH="/opt/venv/bin:$PATH"

# Expose port 5000, which is the port Gunicorn will listen on inside the container.
EXPOSE 5000

# --- Entry Point ---
# The command that runs when a container is started from this image.
# We use Gunicorn, a production-grade WSGI server.
# Gunicorn will run from the WORKDIR (/home/appuser) and find the 'app:create_app()' module.
# The number of workers is now configurable via an environment variable.
ENV GUNICORN_WORKERS 4
CMD ["gunicorn", "--workers", "${GUNICORN_WORKERS}", "--bind", "0.0.0.0:5000", "app:create_app()"]