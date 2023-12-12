# LIBRARIES
import cv2 as cv
import os

# IMPORT THE CLASS HandDetector
import hand_follower as hf

# FOLDER CREATION
name = 'Letter_A'
direction = 'Data'
folder = direction + '/' + name

# IF FOLDER IS NOT CREATED
if not os.path.exists(folder):
	# CREATE DE FOLDER
	os.makedirs(folder)
	print("FOLDER CREATED: ", folder)

# CAMERA READING
cap = cv.VideoCapture(0)
# CHANGE THE RESOLUTION TO 1280x720
cap.set(3, 1280)
cap.set(4, 720)

# DECLARE A COUNTER
counter = 0

# CREATE DETECTOR
detector = hf.HandDetector(reliable_detection = 0.9)

while True:
	# READ FRAMES
	ret, frame = cap.read()

	# EXTRACT HAND INFORMATION
	frame = detector.find_hands(frame, draw = True)

	# OBTAIN THE POSITION OF THE FIRST HAND
	list1, box, hand = detector.find_position(frame, hand_id = 0, draw_points = False, draw_box = False, color = (0, 255, 0))

	# IF THERE IS A HAND
	"""if hand == 1:
		# EXTRACT THE BOX INFORMATION
		x_min, y_min, x_max, y_max = box

		# DO THE RECTANGLE BIGGER TO TAKE THE WHOLE HAND
		x_min = x_min - 40
		y_min = y_min - 40
		x_max = x_max + 40
		y_max = y_max + 40

		# EXTRACT THE HAND
		hand_rectangle = frame[y_min : y_max, x_min : x_max]

		# REDIMENSIONATE
		hand_rectangle = cv.resize(hand_rectangle, (640, 640), interpolation = cv.INTER_CUBIC)

		# SAVE THE HAND IMAGES
		cv.imwrite(folder + "/A_{}.jpg".format(counter), hand_rectangle)

		# INCREMENT THE COUNTER
		counter = counter + 1

		# SHOW THE HAND IMAGE
		cv.imshow("RECORTE", hand_rectangle)"""

	# SHOW FPS
	cv.imshow("HAND DETECTOR", frame)
	# READ THE KEYBOARD
	t = cv.waitKey(1)
	if t == 27 or counter == 150:
		break

cap.release()
cv.destroyAllWindows()