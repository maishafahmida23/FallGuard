import cv2
import cvzone
import math
import time
import pygame
import smtplib
from email.mime.text import MIMEText
from threading import Thread
import torch
from ultralytics import YOLO


# Email Configuration
SENDER_EMAIL = "shutterstorage001@gmail.com"
APP_PASSWORD = "iczu nerm bkam ymiu"  # Use your generated app password
RECIPIENT_EMAIL = "farabi874@gmail.com"

# Initialize pygame for sound playback
pygame.mixer.init()
fall_sound = pygame.mixer.Sound("fall_detected.mp3")  # Ensure the file exists
pygame.mixer.Sound.set_volume(fall_sound, 0.5)  # Reduce CPU load

# Load YOLO model
model = YOLO('yolov5nu.pt')  # Ensure CPU usage on Raspberry Pi

# Load class names (Ensure classes.txt exists)
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Video Capture (Webcam or Video)
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
  # Use V4L2 for better performance
cap.set(cv2.CAP_PROP_FPS, 30)  # Try setting FPS manually

# Tracking previous positions
previous_positions = {}
fall_confirmed = False
fall_start_time = None
sound_played = False
email_sent = False  # Prevent multiple emails
fall_frames = {}

# Fall detection tuning
SPEED_THRESHOLD = 20  
ANGLE_THRESHOLD = 100  
ASPECT_RATIO_THRESHOLD = 2.5  
FALL_CONFIRMATION_FRAMES = 4  
Y_POSITION_THRESHOLD = 100  

def send_email():
    """Send an email alert when a fall is detected."""
    global email_sent
    subject = "Fall Detected Alert!"
    body = "A human fall has been detected by your Raspberry Pi fall detection system."
    
    msg = MIMEText(body)
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        email_sent = True  
    except Exception as e:
        print("Error sending email:", e)

def play_sound():
    """Play the fall detection sound asynchronously."""
    global sound_played
    if not sound_played and not pygame.mixer.get_busy():
        fall_sound.play()
        sound_played = True  

frame_count = 0  # Frame skipping counter

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame
        continue  

    frame = cv2.resize(frame, (640, 640))  # Reduced resolution for faster processing
    results = model(frame)

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]  # Coordinates of the bounding box
        confidence = result.conf[0]  # Confidence level
        class_detect = int(result.cls[0])  # Detected class index

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = math.ceil(confidence * 100)

        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / width
        center_y = (y1 + y2) // 2

        person_id = hash((x1, y1, x2, y2))  # Unique identifier for tracking
        prev_center_y = previous_positions.get(person_id, center_y)
        speed = abs(center_y - prev_center_y)
        previous_positions[person_id] = center_y

        # Body angle
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

        if conf > 80 and classnames[class_detect] == 'person':
            cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
            cvzone.putTextRect(frame, f'Person {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=2)

            # Fall detection conditions
            fall_conditions = (
                aspect_ratio > ASPECT_RATIO_THRESHOLD and  
                speed > SPEED_THRESHOLD and  
                angle > ANGLE_THRESHOLD  
            )

            # Check for significant vertical movement
            if center_y - prev_center_y > Y_POSITION_THRESHOLD:
                fall_conditions = True

            # Fall confirmation logic
            if fall_conditions:
                fall_frames[person_id] = fall_frames.get(person_id, 0) + 1

                if fall_frames[person_id] >= FALL_CONFIRMATION_FRAMES:
                    if not fall_confirmed:
                        fall_confirmed = True
                        fall_start_time = time.time()
                        print("Fall Detected!")

                        # Start sound in a separate thread
                        Thread(target=play_sound).start()

                        if not email_sent:
                            # Send email in a separate thread
                            Thread(target=send_email).start()

            else:
                fall_frames[person_id] = max(0, fall_frames.get(person_id, 0) - 1)

    # Display fall status
    if fall_confirmed:
        cvzone.putTextRect(frame, 'Fall Detected', [50, 50], thickness=2, scale=2, colorR=(0, 0, 255))
    else:
        cvzone.putTextRect(frame, 'No Fall Detected', [50, 50], thickness=2, scale=2, colorR=(0, 255, 0))

    # Reset fall detection after 5 seconds
    if fall_confirmed and time.time() - fall_start_time > 5:
        fall_confirmed = False
        sound_played = False  
        email_sent = False  

    # Show frame using OpenCV
    cv2.imshow('Fall Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
