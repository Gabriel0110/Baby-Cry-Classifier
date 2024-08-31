# Baby-Cry-Classifier
The Baby Cry Classifier is an application designed to classify different types of baby cries using machine learning models. The application listens to audio input, processes it to extract features, and then uses TensorFlow Lite models to predict the type of cry. The application also includes a Streamlit dashboard to visualize the predictions and allows for secure access through an Nginx reverse proxy.  

Models were built using the open-source dataset found here: https://www.kaggle.com/datasets/bhoomikavalani/donateacrycorpusfeaturesdataset  

# Features
- Classification of baby cries (currently set to 4-second audio capture intervals for classification)
- Multiple TensorFlow Lite models are used in a 'stacking' method in an attempt to obtain an accurate prediction.
- Secure access to the Streamlit dashboard using Nginx reverse proxy.
- Automated email notifications for application status changes.
- Easy to set up and run using Docker.  

# Prerequisites
- Docker
- Python 3.11
- Raspberry Pi (recommended but not mandatory)
- Gmail account for email notifications  

# Installation
## Clone the Repository
```
git clone https://github.com/yourusername/baby-cry-classifier.git  
cd baby-cry-classifier
``` 

## Edit necessary variables (IMPORTANT)
In the Dockerfile, add your trusted IP addresses as a space-separated list:  
```
RUN /app/bbc-venv/bin/python /app/generate_nginx_conf.py {IP_ADDRESS(ES)_HERE} # Replace with your trusted IP addresses
```

If you would like email notifications via Gmail API, edit the variables here in send_email.py:  
```
def send_email(subject, body, to="", gmail_user="", gmail_pwd=""):
```

## Build the Docker container
```
docker build -t baby-cry-classifier .
```

## Run the Docker Container
```
docker run --rm --device /dev/snd -d -p 8080:8080 -p 8501:8501 baby-cry-classifier
```

# Configuration
## Nginx Reverse Proxy
The application uses Nginx to secure access to the Streamlit dashboard. The generate_nginx_conf.py script generates the Nginx configuration file based on the trusted IP addresses provided.  

## Email Notifications
Set your email credentials in send_email.py.  
```
def send_email(subject, body, to="youremail@gmail.com", gmail_user="your_gmail_user@gmail.com", gmail_pwd="your_gmail_password"):
```

# Usage
## Start the Application
After running the Docker container, the application will:  

- Start the audio classification service.
- Start the Streamlit dashboard on port 8501.
- Start the Nginx server on port 8080.
- Access the Streamlit Dashboard
- Open your browser and navigate to http://{your-host-ip}:8080.  

## Viewing Predictions
The Streamlit dashboard provides real-time updates on the latest cry predictions and a historical view of all detections.  

## Stopping the Application
To stop the Docker container:  
```
docker ps  # Get the container ID  
docker stop <container_id>
```

# Troubleshooting
## Common Issues
### Audio Input Not Working:
Ensure your audio device is correctly configured and accessible within the Docker container. If no device is detected for some time, it will fail and exit.  

# Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
