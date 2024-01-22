# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /code

# copy requirements.txt from local to image
COPY ./requirements.txt /code/requirements.txt

# install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./camembert_optimized_quantized /code/camembert_optimized_quantized

# Define environment variable
ENV OPTIMIZED_MODEL_PATH=/code/camembert_optimized_quantized

# Run app.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
