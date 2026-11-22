# Aegis — API + Streamlit dashboard in one image (Hugging Face Spaces / Render / any Docker host)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000; the app writes its SQLite DB at runtime
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user . .
RUN chmod +x deploy/start.sh

USER user
ENV HOME=/home/user \
    PORT=7860
EXPOSE 7860

# The API listens on 8000 behind the dashboard, or on $PORT when AEGIS_PUBLIC_UI=console
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '7860') if os.environ.get('AEGIS_PUBLIC_UI') == 'console' else '8000'; urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4)"

CMD ["bash", "deploy/start.sh"]
