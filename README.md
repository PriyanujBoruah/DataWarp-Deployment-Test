# Data Cleaning Platform - Deployment Guide

This document provides instructions for deploying the Data Cleaning Platform using Docker.

## Prerequisites

-   Docker and Docker Compose must be installed on the host server.
-   A shared network drive or a local directory on the host server must be available for persistent data storage.

## Package Contents

-   `data-cleaning-platform-web.tar.gz`: The Docker image for the web application.
-   `data-cleaning-platform-worker.tar.gz`: The Docker image for the background worker.
-   `docker-compose.client.yml`: The Docker Compose file for running the application.
-   `.env.example`: A template for the required environment variables.

## Deployment Steps

**Step 1: Create a Deployment Directory**

Create a dedicated directory on your server and place all the files from this package into it.

**Step 2: Load the Docker Images**

From your terminal, navigate into the deployment directory and run the following commands to load the application images into your local Docker registry:

```bash
docker load < data-cleaning-platform-web.tar.gz
docker load < data-cleaning-platform-worker.tar.gz