
#Created by MediaPipe
#Modified by Augmented Startups 2021
#FaceDetection at 70+ FPS in 5 Minutes
#Watch 5 Minute Tutorial at www.augmentedstartups.info/YouTube
import mediapipe as mp
import cv2
import time
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = mp_face_detection(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        # Cette opération permet de faire la différence entre le vecteur de dimension 128 de
        # tous les visages de notre base avec celui détecté pour récupérer le visage le plus proche
        # du visage détecté. Le résultat de chaque différence est stocké dans vectors
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

        # On a une valeur de tolérence. Donc si la différence est inférieure à 0.6 on considère que c'est le même visage
        # Plus c'est proche de 0 plus les visages sont similaires.
        tolerance = 0.6

        result = []
        # On parcourt le tableau vectors et pour chaque élément du tableau si c'est inférieur
        # à la tolérance on met vrai dans le tableau result sinon on met faux
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)

        if True in result:
            # On récupère l'index du nom dans le tableau de résultat
            first_match_index = result.index(True)

            # On récupère le nom qui correspond à cet index dans le tableau de noms des visages connus
            name = known_face_names[first_match_index]

        else:
            name = "Inconnu"
        face_names.append(name)

    # On dessine les rectangle sur le visage avec le nom en bas etc
    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


#For webcam input:
print("[Allumage de la Webcam]")
cap = cv2.VideoCapture(0)
#For Video input:
#cap = cv2.VideoCapture("1.mp4")



prevTime = 0
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while True:
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    #Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()

# Learn more AI in Computer Vision by Enrolling in our AI_CV Nano Degree:
# https://bit.ly/AugmentedAICVPRO