FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY src/         ./src/
COPY models/      ./models/
COPY data/raw/data.csv ./data/raw/data.csv
COPY params.yaml  ./params.yaml
COPY app.py       ./app.py
RUN mkdir -p reports
EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "app:app"]
