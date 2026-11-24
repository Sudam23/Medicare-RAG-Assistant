# Two stages so the ~350 MB of build tooling and the pip cache never reach the
# runtime image. What ships is the virtualenv, the ONNX model, and the app.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# Bake the embedding model into the image. Downloaded at runtime instead, the
# first visitor to a cold free-tier instance would wait through a ~130 MB fetch
# before getting an answer.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN /opt/venv/bin/python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/models \
    OMP_NUM_THREADS=1 \
    PORT=8000

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data/processed ./data/processed
COPY data/index ./data/index
COPY data/gold ./data/gold
COPY docs ./docs

# Qdrant's embedded mode writes a lock file into its own directory, so the
# index has to be writable by the user the server runs as.
RUN chown -R app:app /app /opt/models
USER app

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/api/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn medicare_rag.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
