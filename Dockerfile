FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
        build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc libffi-dev libssl-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY main.py .
COPY config/ config/

EXPOSE 8000

CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]


# The following commented-out section is an alternative approach using a multi-stage build.
# It works the same way aiming to reduce the final image size by separating the build environment from the runtime environment.

#FROM python:3.13-slim-bookworm AS builder
#
#WORKDIR /app
#
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        gcc \
#        libffi-dev \
#        libssl-dev \
#        build-essential
#
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
#FROM python:3.13-slim-bookworm
#
#WORKDIR /app
#
#RUN adduser --system --group appuser
#RUN chown -R appuser:appuser /app
#
#COPY --from=builder /opt/venv /opt/venv
#
#COPY --chown=appuser:appuser . .
#
#ENV PATH="/opt/venv/bin:$PATH"
#
#USER appuser
#
#EXPOSE 8000
#CMD ["python", "main.py"]
