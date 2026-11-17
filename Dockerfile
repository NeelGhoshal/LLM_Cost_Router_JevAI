# Runs the web demo anywhere that takes a container (Hugging Face Spaces, Fly.io, Koyeb, Render).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN useradd -m app
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY router/ router/
COPY web/ web/
COPY config.yaml config.deploy.yaml ./
COPY data/prompts.jsonl data/prompts.jsonl
COPY results/ results/
USER app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
