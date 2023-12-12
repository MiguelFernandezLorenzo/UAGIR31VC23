# LIBRARIES
import cv2 as cv
import math
import mediapipe as mp
import time

# CREATE A CLASS
class HandDetector():
	# INITIALIZE DETECTION PARAMETERS
	def __init__(self, mode = False, max_hands = 2, model_complexity = 1, reliable_detection = 0.5, reliable_follow = 0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.model_complexity = model_complexity
		self.reliable_detection = reliable_detection
		self.reliable_follow = reliable_follow

		# CREATE THE OBJECTS THAT WILL DRAW AND DETECT THE HANDS
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.reliable_detection, self.reliable_follow)
		self.drawing = mp.solutions.drawing_utils
		self.tip = [4, 8, 12, 16, 20]

	# HANDS FINDER FUNCTION
	def find_hands(self, frame, draw = True):
		img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		self.results = self.hands.process(img_rgb)

		if self.results.multi_hand_landmarks:
			for hand in self.results.multi_hand_landmarks:
				if draw:
					# DRAW THE CONNECTIONS BEETWEEN POINTS
					self.drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

		return frame

	# POSITION FINDER FUNCTION
	def find_position(self, frame, hand_id = 0, draw_points = True, draw_box = True, color = (0, 255, 0)):
		x_list = []
		y_list = []
		box = []
		player = 0
		self.list = []

		if self.results.multi_hand_landmarks:
			my_hand = self.results.multi_hand_landmarks[hand_id]
			proof = self.results.multi_hand_landmarks
			player = len(proof)

			for id, lm in enumerate(my_hand.landmark):
				# EXTRACT FPS DIMENSIONS
				height, width, c = frame.shape
				# CONVERT THE INFORMATION INTO PIXELS
				cx, cy = int(lm.x * width), int(lm.y * height)
				# SAVE THE PIXELS 
				x_list.append(cx)
				y_list.append(cy)
				self.list.append([id, cx, cy])
				
				if draw_points:
					# DRAW A CIRCLE
					cv.circle(frame, (cx, cy), 3, (0, 0, 0), cv.FILLED)

			x_min, x_max = min(x_list), max(x_list)
			y_min, y_max = min(y_list), max(y_list)
			box = x_min, y_min, x_max, y_max

			if draw_box:
				# DRAW THE BOX THAT CONTAIN THE HAND
				cv.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20,y_max - 20), color, 2)

		return self.list, box, player