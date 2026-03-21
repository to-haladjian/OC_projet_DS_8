FROM python:3.13-slim

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install curl for health checks and clean up apt cache
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Give ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Default port: 7860 for HF Spaces, overridable via PORT env var
ENV PORT=7860
EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
