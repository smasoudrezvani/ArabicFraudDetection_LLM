# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (optional, for web services)
# EXPOSE 8000

# Command to run the application (replace with your desired script or framework)
CMD ["python", "src/train.py", "--model", "aubmindlab/bert-base-arabertv2", "--train", "data/processed/train.json", "--test", "data/processed/test.json", "--output", "models/fraud_detector"]
