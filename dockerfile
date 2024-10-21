# Use the official Python image from the Docker Hub
FROM python:3.10

# Set environment variable

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends 

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Specify the default command to run main.py when the container starts
CMD ["python", "main.py"]
