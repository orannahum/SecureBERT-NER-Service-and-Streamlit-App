FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and application code
COPY ner_streamlit_app/requirements.txt .
COPY ner_streamlit_app/app.py .

# Copy the data_processing_transformers.py and functions.py from the parent directory
COPY data_processing_transformers.py .
COPY functions.py .
COPY config.json .
COPY map_tokens.json .
COPY dnrti_to_category.json .

# Copy the pipelines and models directory
COPY pipelines ./pipelines/
COPY models ./models/

# Install required packages
RUN  pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables for Streamlit to run in headless mode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]