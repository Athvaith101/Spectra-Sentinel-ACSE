FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt

# Install all dependencies except PyTorch
RUN grep -v '^torch' /app/backend/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

# Install CPU-only PyTorch (no CUDA/NVIDIA packages)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend /app/backend
COPY python_engine /app/python_engine
COPY models /app/models
COPY configs /app/configs

ENV EW_V6_ROOT=/app
ENV EW_ENGINE_ROOT=/app/python_engine

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8000}"]