FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=5000
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py train.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/
EXPOSE 5000
CMD ["python", "app.py"]
