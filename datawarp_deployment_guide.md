# DataWarp - On-Premise Deployment Guide

Welcome to DataWarp! This document provides instructions for deploying the DataWarp platform on your own server using Docker.

## 1. Prerequisites

Before you begin, please ensure the following software is installed on your host server:

*   **Docker:** [Official Installation Guide](https://docs.docker.com/engine/install/)
*   **Docker Compose:** [Official Installation Guide](https://docs.docker.com/compose/install/)

A shared network drive or a local directory on the host server must be available for persistent data storage.

## 2. Package Contents

The deployment package you received contains the following files:

*   `datawarp-app.tar.gz`: The Docker image for the application.
*   `docker-compose.client.yml`: The Docker Compose file for running the application.
*   `.env.example`: A template for the required environment variables.

## 3. Deployment Steps

Follow these steps to get the application up and running.

### Step 1: Create a Deployment Directory
Create a dedicated directory on your server (e.g., `/opt/datawarp/`) and extract all the files from the provided package into it.

### Step 2: Configure the Environment
The application is configured using an `.env` file.

1.  Make a copy of the template file:
    ```bash
    cp .env.example .env
    ```

2.  Open the `.env` file with a text editor and fill in the required values:
    *   `FLASK_SECRET_KEY`: Generate a long, random string for session security.
    *   `SHARED_FOLDER_PATH`: **This is critical.** Set this to the absolute path of the directory on your host server where all application data (datasets, user files, etc.) will be stored. This directory must be writable by the user running Docker.
    *   `SUPABASE_URL`: Your Supabase project URL.
    *   `SUPABASE_KEY`: Your Supabase `anon` key.
    *   `SUPABASE_SERVICE_KEY`: Your Supabase `service_role` key.
    *   `GUNICORN_WORKERS`: (Optional) The number of web server processes. Defaults to 4, which is suitable for most servers with 2-4 CPU cores.

### Step 3: Load the Docker Image
From your terminal, navigate into the deployment directory and run the following command to load the application image into your local Docker registry:

```bash
docker load < datawarp-app.tar.gz
```

### Step 4: Start the Application
Start all application services in the background, scaling the background worker service to 5 instances for optimal performance.

```bash
docker-compose -f docker-compose.client.yml up -d --scale worker=5```
The application is now running. It may take a minute for all services to initialize fully.

### Step 5: Accessing the Application
You can now access the DataWarp platform in a web browser at:

`http://<your-server-ip>:5000`

## 4. First-Time Setup: Creating the First Admin User

The application is running, but no users exist yet. You must run a command-line script to create the first organization and its administrator.

From your deployment directory, run the following command, replacing the placeholder values with your desired details:

```bash
docker-compose -f docker-compose.client.yml exec web python create_user.py --org "Your Organization Name" --email "your-admin-email@company.com" --password "ChooseASecurePassword"
```
Once this command completes successfully, you can log in with the email and password you just created.

## 5. Application Management

Here are some basic commands to manage the application, run from your deployment directory:

*   **View Logs:** To see the real-time logs from all services:
    ```bash
    docker-compose -f docker-compose.client.yml logs -f
    ```

*   **Stop the Application:** To stop all running services:
    ```bash
    docker-compose -f docker-compose.client.yml down
    ```

*   **Restart the Application:** To restart after stopping (remember to include the scale flag):
    ```bash
    docker-compose -f docker-compose.client.yml up -d --scale worker=5
    ```

## 6. Data Backup

All persistent application data, including uploaded datasets and user-generated files, is stored in the directory you specified for `SHARED_FOLDER_PATH` in your `.env` file.

**It is your responsibility to implement a regular backup strategy for this directory.**

## 7. Updating the Application

When a new version of DataWarp is provided, the update process will typically involve:
1.  Loading the new `.tar.gz` image file.
2.  Stopping the current application with `docker-compose -f docker-compose.client.yml down`.
3.  Restarting the application with the new image and correct scaling: `docker-compose -f docker-compose.client.yml up -d --scale worker=5`.

Detailed instructions will be provided with each update.