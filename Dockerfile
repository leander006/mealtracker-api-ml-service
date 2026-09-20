FROM python:3.11-slim

# libgl1/libglib2.0-0 are required by opencv-python-headless at import time,
# even though "headless" implies no GUI deps - this is a common gotcha.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set to "true" to also install torch/ultralytics for DETECTOR_MODE=real.
# Default false keeps local/mock builds fast and avoids downloading torch
# at all when you don't need it yet:
#   docker compose build --build-arg INSTALL_REAL_DETECTOR=true
ARG INSTALL_REAL_DETECTOR=false

COPY requirements.txt requirements-real.txt ./
RUN pip install --no-cache-dir --default-timeout=180 --retries 5 -r requirements.txt
RUN if [ "$INSTALL_REAL_DETECTOR" = "true" ]; then \
      pip install --no-cache-dir --default-timeout=180 --retries 5 -r requirements-real.txt; \
    fi

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
