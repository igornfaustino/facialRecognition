import os
import cv2
import face_utils

def prepare_training_data(data_folder_path):
	faces = []
	labels = []
	dirs = os.listdir(data_folder_path)

	for dir_name in dirs:
		label = int(dir_name.replace("s", ""))

		subject_dir_path = data_folder_path + "/" + dir_name
		subject_images_names = os.listdir(subject_dir_path)

		for image_name in subject_images_names:
			image_path = subject_dir_path + "/" + image_name
			image = cv2.imread(image_path)

			# cv2.imshow("training", image)
			# cv2.waitKey(100)

			face, rect = face_utils.detect_face(image)

			if face is not None:
				faces.append(face)
				labels.append(label)
		
			# cv2.destroyAllWindows()
			# cv2.waitKey(1)
			# cv2.destroyAllWindows()
 
	return faces, labels