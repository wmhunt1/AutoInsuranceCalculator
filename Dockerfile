# We start with the official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# --- FIX: Install System Dependencies for Compilation ---
# These packages provide the necessary C/C++ compilers, the Fortran compiler, and the 
# high-performance linear algebra library (OpenBLAS) required by scikit-learn and numpy.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
# --- END FIX ---

# Copy the requirements file into the working directory
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your Flask application runs on
EXPOSE 5000

# Define the production command using Gunicorn
# FIX: Using 'python -m gunicorn' ensures the executable is found reliably within the Python environment.
CMD ["python", "-m", "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]