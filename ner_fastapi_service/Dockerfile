FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and application code
COPY ner_fastapi_service/requirements.txt .
COPY ner_fastapi_service/main.py .

# Copy the data_processing_transformers.py and functions.py from the parent directory
COPY data_processing_transformers.py .
COPY functions.py .
COPY config.json .

# Copy the pipelines and models directory
COPY pipelines ./pipelines/
COPY models ./models/

# Install required packages
RUN  pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
