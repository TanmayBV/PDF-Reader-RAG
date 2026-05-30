# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies:
# - tesseract-ocr: Needed for pytesseract OCR of scanned pages
# - libgl1-mesa-glx & libglib2.0-0: Needed for OpenCV and PyMuPDF
# - build-essential: Needed for building native extensions if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV FAISS_STORE_PATH=/workspace/faiss_store
ENV PYTHONPATH=/workspace

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Launch server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/workspace"]
