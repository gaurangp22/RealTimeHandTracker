import random
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import time

# Initialize webcam
camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

# Create hand detector instance
hand_detector = HandDetector(detectionCon=0.8, maxHands=1)

# Distance to centimeter conversion function
# x: raw distance, y: equivalent value in cm
raw_distances = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
cm_values = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coefficients = np.polyfit(raw_distances, cm_values, 2)  # y = Ax^2 + Bx + C

# Game parameters
target_x, target_y = 250, 250
circle_color = (255, 0, 255)
tap_counter = 0
player_score = 0
game_start_time = time.time()
game_duration = 20  # seconds

# Game loop
while True:
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if time.time() - game_start_time < game_duration:
        detected_hands = hand_detector.findHands(frame, draw=False)

        if detected_hands:
            landmarks_list = detected_hands[0]['lmList']
            x, y, w, h = detected_hands[0]['bbox']
            thumb_x, thumb_y = landmarks_list[5]
            pinky_x, pinky_y = landmarks_list[17]

            # Calculate distance between thumb and pinky
            distance = int(math.sqrt((pinky_y - thumb_y) ** 2 + (pinky_x - thumb_x) ** 2))
            A, B, C = coefficients
            distance_cm = A * distance ** 2 + B * distance + C

            if distance_cm < 40:
                if x < target_x < x + w and y < target_y < y + h:
                    tap_counter = 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(frame, f'{int(distance_cm)} cm', (x + 5, y - 10))

        if tap_counter:
            tap_counter += 1
            circle_color = (0, 255, 0)
            if tap_counter == 3:
                target_x = random.randint(100, 1100)
                target_y = random.randint(100, 600)
                circle_color = (255, 0, 255)
                player_score += 1
                tap_counter = 0

        # Draw the target circle
        cv2.circle(frame, (target_x, target_y), 30, circle_color, cv2.FILLED)
        cv2.circle(frame, (target_x, target_y), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame, (target_x, target_y), 20, (255, 255, 255), 2)
        cv2.circle(frame, (target_x, target_y), 30, (50, 50, 50), 2)

        # Game HUD display
        cvzone.putTextRect(frame, f'Time: {int(game_duration - (time.time() - game_start_time))}',
                           (1000, 75), scale=3, offset=20)
        cvzone.putTextRect(frame, f'Score: {str(player_score).zfill(2)}', (60, 75), scale=3, offset=20)
    else:
        cvzone.putTextRect(frame, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
        cvzone.putTextRect(frame, f'Your Score: {player_score}', (450, 500), scale=3, offset=20)
        cvzone.putTextRect(frame, 'Press R to restart', (460, 575), scale=2, offset=10)

    # Show the processed image
    cv2.imshow("Game Frame", frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        game_start_time = time.time()
        player_score = 0
