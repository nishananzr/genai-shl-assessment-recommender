FROM python:3.12-slim 
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY ./data ./data
COPY ./src ./src

ENV PORT 8080
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "4", "--timeout", "120", "src.api:app"]