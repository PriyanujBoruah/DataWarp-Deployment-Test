# DataWarp - Developer & Maintainer Guide

This document is the internal guide for developing, maintaining, and packaging the DataWarp application.

## 1. Project Architecture

The application consists of three main services orchestrated by Docker Compose:
*   **`web`**: The Flask/Gunicorn web server that serves the UI and handles API requests.
*   **`worker`**: An RQ `SimpleWorker` that processes long-running background jobs (e.g., ML benchmarking, report generation, data imputation).
*   **`redis`**: The message broker that manages the job queues between the `web` and `worker` services.

## 2. Local Development Setup

Follow these steps to run the full application on your local machine for development.

### Prerequisites
*   Python 3.11
*   Docker Desktop
*   A Python virtual environment tool (`venv`)

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd Data-Cleaning-Platform
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create the venv
    python -m venv .venv

    # Activate it (PowerShell)
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies:**
    Install `pip-tools` and then compile your `requirements.in` to install all packages.
    ```bash
    pip install pip-tools
    pip-compile requirements.in > requirements.txt
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Copy the example `.env` file and fill in your development keys.
    ```powershell
    copy .env.example .env
    ```
    Ensure `SHARED_FOLDER_PATH` in the `.env` file points to a real directory on your machine (e.g., a `local_shared_data` folder in the project root).

5.  **Run with Docker Compose:**
    This command will build the images, start all services in detached mode, and scale the worker service to 5 replicas.
    ```bash
    docker-compose up -d --scale worker=5
    ```

6.  **Access the Application:**
    The application will be available at `http://localhost:5000`.

## 3. Dependency Management

This project uses `pip-tools` to ensure deterministic builds. **Do not manually edit `requirements.txt`.**

*   **To Add/Update a Package:** Add or change the package version in `requirements.in`.
*   **To Generate `requirements.txt`:** After changing `requirements.in`, you **must** run the following command to regenerate the lock file.

    ```bash
    # This command ensures dependencies are compatible with the Docker environment's Python version
    pip-compile --python-version 3.11 requirements.in > requirements.txt
    ```
*   **OS-Specific Packages:** Windows-only packages like `pywin32` must be excluded. They are managed in the `pyproject.toml` file under the `[tool.pip-tools]` section.

## 4. Creating the Client Deployment Package

This is the process for building the final `.zip` file to send to a client.

1.  **Ensure Your Image is Up-to-Date:** Build the image using your final, committed code.
    ```bash
    docker-compose build
    ```

2.  **Create a Package Directory:**
    ```bash
    mkdir deployment_package
    ```

3.  **Save the Docker Image:**
    Identify the final image name using `docker images` (it should be `datacleaningplatform-web`). We only need to save this one image as it's used for both the `web` and `worker` services.

    *   **On Windows (using PowerShell & WSL):**
        ```powershell
        # 1. Save the uncompressed .tar file
        docker save datacleaningplatform-web:latest -o deployment_package/datawarp-app.tar

        # 2. Open WSL to compress it
        wsl
        # IMPORTANT: Change this to your project path if it differs
        cd /mnt/c/path/to/your/project/Data-Cleaning-Platform/ 
        gzip deployment_package/datawarp-app.tar
        exit
        ```

    *   **On Linux/macOS:**
        ```bash
        docker save datacleaningplatform-web:latest | gzip > deployment_package/datawarp-app.tar.gz
        ```

4.  **Copy Client Configuration Files:**
    ```powershell
    # On Windows PowerShell
    copy docker-compose.client.yml deployment_package\
    copy .env.example deployment_package\
    ```

5.  **Create the Final Zip Archive:**
    Right-click the `deployment_package` folder and choose "Send to > Compressed (zipped) folder". Name it something like `DataWarp_Deployment_v1.0.zip`. This is the final deliverable.

## 5. Running Admin Scripts

The super-admin scripts (`create_user.py`, `reset_password.py`) must be run *inside* a running container to access the correct environment variables.

*   **Example: Creating a new organization and user:**
    ```bash
    docker-compose exec web python create_user.py --org "Client Org Name" --email "admin@client.com" --password "a-secure-password"
    ```