FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY . .

RUN pip install --no-cache-dir .

EXPOSE 6000

# Run the application
CMD ["fastapi", "run", "app/main.py", "--port", "6000", "--host", "0.0.0.0"]
