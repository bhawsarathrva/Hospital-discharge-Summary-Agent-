# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/discharge_agent

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy requirements file first for caching
COPY requirements.txt ./

# Install python dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Ensure outputs directory exists
RUN mkdir -p outputs

# Default command to run the multi-agent graph pipeline
CMD ["python", "app.py"]
