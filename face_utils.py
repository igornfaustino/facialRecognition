import cv2
import image_utils

def detect_face(img):
	"""
		Extract a face

		get an image and return a face

		if not found a face, return nonde

		*** Only supports images with just one face
	"""

	# Convert an image to a gray scale
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# Load Cascade
	face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
	
	# Detect an list of faces in the image
	faces = face_cascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
	
	# If no face, return none
	if (len(faces) == 0):
		return None, None
	
	# Get just one face (this code just supports one face)
	(x, y, w, h) = faces[0]
	
	# return only the face part of the image
	return grayImage[y:y+w, x:x+h], faces[0]

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the 
# subject
def predict(test_img, face_recognizer, subjects):
	# make a copy of the image as we don't want to change original image
	img = test_img.copy()
	if(img is not None):
		# detect face from the image
		face, rect = detect_face(img)
		if face is not None:
			# predict the image using our face recognizer 
			label = face_recognizer.predict(face)[0]
			# get name of respective label returned by face recognizer
			label_text = subjects[label]
			
			# draw a rectangle around face detected
			image_utils.draw_rectangle(img, rect)
			# draw name of predicted person
			image_utils.draw_text(img, label_text, rect[0], rect[1]-5)
	
	return img