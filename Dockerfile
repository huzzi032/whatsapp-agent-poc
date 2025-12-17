# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt with longer timeout
RUN pip install --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable for port (Azure might set PORT)
ENV PORT=8000

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT