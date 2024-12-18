FROM python3.12.0-alpine3.14

# Set the working directory
WORKDIR /app

# Copy the requierements file
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . /app
