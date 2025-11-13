
# Use official Python slim image
FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install system deps for torch (minimal) - user may need to adapt for GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
