FROM python:3.8-slim

WORKDIR /app

# Install system deps for opencv and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy repo (optimize by copying only needed paths; for simplicity copy all)
COPY . /app

# Install requirements
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn pydantic PyJWT requests python-multipart

# Install CPU-only PyTorch and torchvision (needed by HRNet test script)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==1.13.1+cpu torchvision==0.14.1+cpu

EXPOSE 8000

CMD ["uvicorn", "api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]


