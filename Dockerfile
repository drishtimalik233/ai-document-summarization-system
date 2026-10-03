FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# OLLAMA_HOST must point to a reachable Ollama server (see README deployment section)
ENV OLLAMA_HOST=http://host.docker.internal:11434
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
