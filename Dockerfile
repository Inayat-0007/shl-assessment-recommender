FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Start FastAPI backend
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
