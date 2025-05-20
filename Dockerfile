# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port your app runs on (if using Flask or similar)
EXPOSE 5000

# Command to run your app (adjust if you use main.py or app1.py)
CMD ["python", "app1.py"]
