import cv2
import image_utils

def detect_face(img):
	"""
		Extract a face

		get an image and return a face

		if not found a face, return none
	"""

	# Convert an image to a gray scale
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# Load Cascade
	face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
	
	# Detect an list of faces in the image
	faces = face_cascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5)
	
	# If no face, return none
	if (len(faces) == 0):
		return None, None
	
	# get all faces
	grayFaces = []
	for face in faces:
		(x, y, w, h) = face
		grayFaces.append(grayImage[y:y+w, x:x+h])
	
	# return only the face part of the image
	return grayFaces, faces

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the 
# subject
def predict(test_img, face_recognizer, subjects):
	# make a copy of the image as we don't want to change original image
	img = test_img.copy()
	if(img is not None):
		# detect face from the image
		faces, rects = detect_face(img)
		if faces is None:
			return img

		for i in range(len(faces)):
			if faces[i] is not None:
			# predict the image using our face recognizer
				label, conf = face_recognizer.predict(faces[i])
				
				if (conf < 50):
					# get name of respective label returned by face recognizer
					label_text = subjects[label]
				else:
					label_text = "Unknown"
				
				# draw a rectangle around face detected
				image_utils.draw_rectangle(img, rects[i])
				# draw name of predicted person
				image_utils.draw_text(img, label_text, rects[i][0], rects[i][1]-5)
	
	return img