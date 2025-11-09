# Use a slim, stable Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# --- 1. Install System Dependencies for Compilation ---
# These packages (gcc, g++, gfortran, libopenblas-dev) are crucial for 
# successfully compiling heavy Python packages like numpy and scikit-learn
# on the Debian-based slim image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
# --- END SYSTEM DEPENDENCIES ---

# Copy the requirements file and install Python dependencies
# This step must run after the system dependencies are installed.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py, Telematicsdata.csv, etc.)
COPY . .

# Expose the port your Flask application runs on
EXPOSE 5000

# --- 2. Robust Startup Command (Shell Form) ---
# Using the SHELL form (without brackets) is the most reliable way to execute 
# the 'gunicorn' command and ensure its path is correctly resolved on startup.
CMD gunicorn -w 4 -b 0.0.0.0:5000 app:app