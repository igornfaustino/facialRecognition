# Author: Igor N Faustino
# Mail: igornfaustino@gmail.com
# TODO Next: Support to many faces

import cv2
import sys
import data_utils
import face_utils
import numpy as np

# Get user values
try:
    train_path = sys.argv[1]
except IndexError as ex:
    print("please enter a training folder path")
    exit(-1)
except:
    exit(-1)

subjects = ["", "Igor"]

print("preparing data....")
faces, labels = data_utils.prepare_training_data(train_path)
print("done...")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

print("training....")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print("Done...")

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    image = face_utils.predict(frame, face_recognizer, subjects)
    
    # Display the resulting frame
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()