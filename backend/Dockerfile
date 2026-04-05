# Use a standard full Python image (more stable than slim for this)
FROM python:3.11

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend folder into the container
COPY . .

# Ensure environment variables are set
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# The app listens on port 8080
EXPOSE 8080

# Start gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "300", "--workers", "1", "--threads", "2"]
