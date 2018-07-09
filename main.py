import cv2
import sys

# Get user values
try:
    cascade_path = sys.argv[1]
except IndexError as ex:
    print("please enter a cascade path")
    exit(-1)
except:
    exit(-1)

# create cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# init video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # detect face in the images
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.6,
        minNeighbors=5,
        minSize=(30, 30),
        # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()