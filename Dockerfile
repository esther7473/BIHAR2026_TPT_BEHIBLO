FROM python:3.13-slim-trixie

WORKDIR /app

# ✅ dépendances d'abord (layer cachée)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ code source après (layer changeante)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]