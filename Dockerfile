FROM python:3.13-slim AS base
WORKDIR /app
COPY config.yml .

FROM base AS serving
COPY requirements/serving.txt .
RUN pip install --no-cache-dir -r serving.txt
COPY src/common/            ./src/common/
COPY src/monitoring/        ./src/monitoring/   
COPY src/data/              ./src/data/
COPY api/                   ./api/
COPY config.yml             ./config.yml       
COPY src/inference/get_run.py ./src/inference/get_run.py  
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS streamlit
COPY requirements/streamlit.txt .
RUN pip install --no-cache-dir -r streamlit.txt
COPY src/common/     ./src/common/
COPY config.yml             ./config.yml       
COPY monitoring/app_streamlit.py      ./app/app_streamlit.py 
EXPOSE 8501
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]