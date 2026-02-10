# Use the stable version we just validated
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Kaleido/Plotly
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from ROOT
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH so src/ modules are discoverable
ENV PYTHONPATH=/app

# Command to run the engine
CMD ["python", "src/main.py"]
