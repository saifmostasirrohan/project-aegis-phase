# Use a pinned, lightweight base image
FROM python:3.12-slim

# Set shell execution configurations for safety
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install minimal system utilities needed for compiling C-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Leverage Layer Caching for backend dependencies
COPY requirements.backend.txt .

# Force pip to fetch lightweight CPU versions of torch binaries for the backend
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.backend.txt

# Create an unprivileged user for runtime security
RUN mkdir -p /app/data /app/chroma_db \
    && useradd --create-home --uid 1000 appuser \
    && chown -R 1000:1000 /app
USER 1000:1000

# Copy backend source code with correct file ownership
COPY --chown=1000:1000 . .

# Expose backend API port
EXPOSE 8000
CMD ["uvicorn", "capstone.api:app", "--host", "0.0.0.0", "--port", "8000"]
