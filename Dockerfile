# ── Base commune ──────────────────────────
FROM python:3.13-slim-trixie AS base
WORKDIR /app
COPY config.yml .

# ── API (serving) ─────────────────────────
FROM base AS serving
COPY requirements/serving.txt .
RUN pip install --no-cache-dir -r serving.txt
COPY api/        ./api/
COPY src/common/ ./src/common/
COPY config.yml .config.yml
COPY src/data/   ./src/data/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# # ── Training ──────────────────────────────
# FROM base AS training
# COPY requirements/training.txt .
# RUN pip install --no-cache-dir -r training.txt
# COPY src/common/   ./src/common/
# COPY src/data/     ./src/data/
# COPY src/training/ ./src/training/
# CMD ["python", "-m", "src.training.train"]

# ── Monitoring ────────────────────────────
FROM base AS monitoring
COPY requirements/monitoring.txt .
RUN pip install --no-cache-dir -r monitoring.txt
COPY src/common/    ./src/common/
COPY src/monitoring/ ./src/monitoring/
CMD ["python", "-m", "src.monitoring.monitor"]