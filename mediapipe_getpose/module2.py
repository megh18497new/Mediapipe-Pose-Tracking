import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
img = cap.read()
def drawpose_video(img):
  with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    img.flags.writeable = True
    results = pose.process(img)

    # Draw the pose annotation on the image.
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2 = np.zeros((height,width,3), np.uint8)
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    result = cv2.imshow('Pose Estimation', img2)
    return result
cap.release()
drawpose_video(img)

