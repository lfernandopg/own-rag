FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd -r dev -g 1000 \
    && useradd -m -r -g dev -u 1000 -d /home/dev -s /bin/bash dev
# Instalar dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/.venv
RUN mkdir -p /app/data
RUN chown -R dev:dev /app/
USER dev
# Instalar librerías de Python (Optimizadas para CPU)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Comando por defecto que mantiene vivo el contenedor
CMD ["tail", "-f", "/dev/null"]