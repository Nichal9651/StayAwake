import cv2
import dlib
from scipy.spatial import distance
from pygame import mixer
import time

# Load the sound
mixer.init()
mixer.music.load("alarm.wav")

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark points
(lStart, lEnd) = (42, 48)  # Right eye
(rStart, rEnd) = (36, 42)  # Left eye

cap = cv2.VideoCapture(0)
sleep_counter = 0
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eyes
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            sleep_counter += 1
            if sleep_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                mixer.music.play()
        else:
            sleep_counter = 0
            mixer.music.stop()

    cv2.imshow("StayAwake - Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
