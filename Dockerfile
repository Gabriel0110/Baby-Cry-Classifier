# Use the Ubuntu 22.04 base image
FROM ubuntu:22.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    build-essential \
    curl \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libatlas-base-dev \
    ffmpeg \
    alsa-utils \
    alsa-oss \
    alsa-tools \
    pulseaudio \
    openjdk-11-jdk \
    unzip \
    zip \
    clang \
    libhdf5-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for the project
RUN python3.11 -m venv /app/bbc-venv

# Upgrade pip to the latest version within the virtual environment
RUN /app/bbc-venv/bin/pip install --upgrade pip

# Install Tensorflow
RUN /app/bbc-venv/bin/pip install tensorflow

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies within the virtual environment
RUN /app/bbc-venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Generate nginx.conf using the script (ADD YOUR IP ADDRESS(ES)) - this is for the reverse proxy
RUN /app/bbc-venv/bin/python /app/generate_nginx_conf.py {IP_ADDRESS(ES)_HERE} # Replace with your trusted IP addresses

# Remove default nginx configuration
RUN rm /etc/nginx/sites-enabled/default

# Copy generated nginx configuration file
RUN cp /app/nginx.conf /etc/nginx/sites-available/streamlit
RUN ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/

# Start PulseAudio
RUN pulseaudio --start

# Expose nginx port
EXPOSE 8080

# Expose the Streamlit port
EXPOSE 8501

# Activate the virtual environment and start the application and Streamlit dashboard
CMD ["/bin/bash", "-c", "service nginx start && source /app/bbc-venv/bin/activate && /app/bbc-venv/bin/python main.py & HOST_IP=$(hostname -I | awk '{print $1}') && /app/bbc-venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 && echo \"Server started. Access streamlit site here: http://$HOST_IP:8501/\""]
