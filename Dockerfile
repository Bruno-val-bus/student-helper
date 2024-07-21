FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# install 3rd party libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the directory contents into the container at /app
COPY . .

CMD ["python", "./main.py"]