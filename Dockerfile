# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app/

# Copy the current directory contents into the container at /app
COPY app/ /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to be accessible outside the container
EXPOSE 8000

# Run mlops_pipeline.py when the container launches
CMD ["python", "mlops_pipeline.py"]
