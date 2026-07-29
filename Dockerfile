FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces expects the app on port 7860.
# Note: app.py is the Gradio entrypoint (run via `python app.py`), not an ASGI app.
# This Dockerfile serves the FastAPI + JSON API variant (app_fastapi.py) for
# non-HF Docker hosts (Render etc.) per README.
EXPOSE 7860
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]
