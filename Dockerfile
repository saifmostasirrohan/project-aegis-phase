FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY phase4_research/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m appuser

COPY --chown=appuser:appuser phase4_research ./phase4_research

RUN chown -R appuser:appuser /app
USER appuser

WORKDIR /app/phase4_research

EXPOSE 8000
CMD ["uvicorn", "capstone.api:app", "--host", "0.0.0.0", "--port", "8000"]
