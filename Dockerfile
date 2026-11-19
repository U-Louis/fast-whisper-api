ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-whisper.txt ./
# pip's bundled CA list rejects some networks.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-whisper.txt

RUN useradd --create-home --uid 10001 whisper
USER whisper
ENV HOME=/home/whisper \
    WHISPER_MODEL=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_CPU_THREADS=4 \
    HF_HOME=/home/whisper/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# Weights are part of the image. A later start does not download them.
RUN python -c "from faster_whisper import download_model; download_model('small')"

COPY --chown=whisper:whisper app ./app
COPY --chown=whisper:whisper whisper-api.conf ./whisper-api.conf

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8080"]
