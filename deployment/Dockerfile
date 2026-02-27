FROM python:3.13.7

WORKDIR /app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content
COPY . .

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

ENTRYPOINT ["python" "src/predict.py"]
