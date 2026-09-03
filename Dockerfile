# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (for faster builds)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY main.py .
COPY train_model.py .
COPY fraud_model.pkl .
COPY feature_columns.pkl .

# Expose port 8000 (the API port)
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]