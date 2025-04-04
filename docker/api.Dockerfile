FROM python:3.11-slim

WORKDIR /app
# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY app/requirements.txt .
RUN pip install -r requirements.txt
COPY app/model/ model/
CMD ["python", "model/serve.py"]