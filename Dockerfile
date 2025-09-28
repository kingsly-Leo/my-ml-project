# Use small, official Python image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port used by Flask
EXPOSE 5000

# Use non-root user
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Copy files for runtime (ownership)
COPY --chown=appuser:appuser . .

# Start the Flask app
CMD ["python", "app.py"]
