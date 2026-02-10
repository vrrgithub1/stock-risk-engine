# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the root to the container
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY .. .

# Command to run the engine (Adjust this to your actual main script name)
CMD ["python", "src/main.py"]
