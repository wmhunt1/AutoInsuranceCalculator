FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /usr/src/app

# 2. Install Build Dependencies: These are critical for compiling binary packages (like pandas/numpy)
# 'apk add' installs tools like gcc (C compiler) and musl-dev (C library headers)
RUN apk add --no-cache gcc g++ musl-dev

# 3. Copy and Install Python Dependencies
# Copy requirements first to leverage Docker's layer caching
COPY requirements.txt .

# Install all Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Cleanup Build Dependencies
# Remove the heavy build tools to minimize the final container image size
RUN apk del gcc musl-dev

# 5. Copy Application Code and Data Files
# Copy the rest of the current directory content into the container
COPY . .

# 6. Expose Port
# Inform Docker that the container will listen on this port
EXPOSE 5000

# 7. Define Environment Variables
# Essential for running Flask/Gunicorn
ENV FLASK_APP=app.py
# Set host to 0.0.0.0 to make the service accessible outside the container
ENV FLASK_RUN_HOST=0.0.0.0

# 8. Production Command (CMD)
# This is the final instruction: Run the application using Gunicorn
# 'app:app' refers to the Flask instance named 'app' inside the file 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]