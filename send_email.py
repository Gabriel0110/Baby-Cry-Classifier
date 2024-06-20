import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ADD YOUR EMAIL CREDENTIALS HERE (USING GMAIL API)
def send_email(subject, body, to="", gmail_user="", gmail_pwd=""):
    if not gmail_user or not gmail_pwd or not to:
        print("[!] EMAIL ERROR: Please provide email credentials in send_email.py for email capabilities. No email will be sent.")
        return
    
    # Prepare the email
    msg = MIMEMultipart()
    msg['From'] = gmail_user
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        text = msg.as_string()
        server.sendmail(gmail_user, to, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
        
        
def async_send_email(*args, **kwargs):
    email_thread = threading.Thread(target=send_email, args=args, kwargs=kwargs)
    email_thread.start()