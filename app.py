# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

KNOWN_FACES_DIR = "known_faces"
FACE_RECOGNIZER_PATH = "face_recognizer.yml"
FACE_LABELS_PATH = "face_labels.pickle"
FACE_SIZE = (200, 200)
RECOGNITION_THRESHOLD = 75.0
CAMERA_INDEXES = (0, 1, 2, 3)

def detect_face_boxes(frame, faceNet, min_confidence=0.5):
	if frame is None:
		return []

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	boxes = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > min_confidence:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			if endX > startX and endY > startY:
				boxes.append((startX, startY, endX, endY))

	return boxes

def prepare_face_for_recognition(face):
	gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	return cv2.resize(gray, FACE_SIZE)

def largest_face_box(frame, faceNet):
	boxes = detect_face_boxes(frame, faceNet)
	if len(boxes) == 0:
		return None

	return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

def create_face_recognizer():
	if not hasattr(cv2, "face"):
		print("[WARN] Face recognition needs opencv-contrib-python.")
		print("[WARN] Install it with: pip install opencv-contrib-python")
		return None

	return cv2.face.LBPHFaceRecognizer_create()

def train_face_recognizer(faceNet):
	recognizer = create_face_recognizer()
	if recognizer is None:
		return (None, {})

	if not os.path.isdir(KNOWN_FACES_DIR):
		print("[WARN] known_faces folder was not found. Skipping face recognition.")
		return (None, {})

	faces = []
	labels = []
	label_names = {}
	current_label = 0

	for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
		person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
		if not os.path.isdir(person_dir):
			continue

		person_faces = 0
		label_names[current_label] = person_name

		for filename in os.listdir(person_dir):
			image_path = os.path.join(person_dir, filename)
			image = cv2.imread(image_path)
			if image is None:
				continue

			box = largest_face_box(image, faceNet)
			if box is None:
				continue

			(startX, startY, endX, endY) = box
			face = image[startY:endY, startX:endX]
			faces.append(prepare_face_for_recognition(face))
			labels.append(current_label)
			person_faces += 1

		if person_faces > 0:
			print("[INFO] loaded {} face image(s) for {}".format(
				person_faces, person_name))
			current_label += 1
		else:
			del label_names[current_label]

	if len(faces) == 0:
		print("[WARN] No usable known face images found. Skipping face recognition.")
		return (None, {})

	recognizer.train(faces, np.array(labels))
	recognizer.save(FACE_RECOGNIZER_PATH)

	with open(FACE_LABELS_PATH, "wb") as f:
		pickle.dump(label_names, f)

	print("[INFO] trained face recognizer for {} person(s)".format(
		len(label_names)))
	return (recognizer, label_names)

def load_face_recognizer(faceNet):
	recognizer = create_face_recognizer()
	if recognizer is None:
		return (None, {})

	if os.path.exists(FACE_RECOGNIZER_PATH) and os.path.exists(FACE_LABELS_PATH):
		recognizer.read(FACE_RECOGNIZER_PATH)
		with open(FACE_LABELS_PATH, "rb") as f:
			label_names = pickle.load(f)

		print("[INFO] loaded trained face recognizer")
		return (recognizer, label_names)

	return train_face_recognizer(faceNet)

def recognize_face(face, recognizer, label_names):
	if recognizer is None or len(label_names) == 0:
		return ("Unknown", 0)

	face = prepare_face_for_recognition(face)
	(label_id, confidence) = recognizer.predict(face)

	if confidence > RECOGNITION_THRESHOLD:
		return ("Unknown", confidence)

	person_name = label_names.get(label_id, "Unknown")
	return (person_name, confidence)

def open_camera():
	for camera_index in CAMERA_INDEXES:
		cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
		if not cap.isOpened():
			cap.release()
			continue

		ok, frame = cap.read()
		if ok and frame is not None:
			print("[INFO] using camera index {}".format(camera_index))
			return cap

		cap.release()

	print("[ERROR] Could not open any camera. Check camera permissions or connect a webcam.")
	return None

def detect_and_predict_mask(frame, faceNet, maskNet):
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for (startX, startY, endX, endY) in detect_face_boxes(frame, faceNet):
		# extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
		face = frame[startY:endY, startX:endX]
		if face.size == 0:
			continue

		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)

		# add the face and bounding boxes to their respective
		# lists
		faces.append(face)
		locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

class FaceMaskDetectorGUI:
	def __init__(self, root):
		self.root = root
		self.root.title("Face Mask Detection System")
		self.root.geometry("1200x800")
		self.root.state('zoomed')  # Fullscreen on Windows
		
		# Global variables
		self.vs = None
		self.is_running = False
		self.thread = None
		
		# Load models
		print("[INFO] Loading models...")
		prototxtPath = r"face_detector\deploy.prototxt"
		weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
		self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
		self.maskNet = load_model("mask_detector.model")
		self.faceRecognizer, self.faceLabelNames = load_face_recognizer(self.faceNet)
		print("[INFO] Models loaded successfully!")
		
		# Create GUI
		self.create_gui()
		
	def create_gui(self):
		# Main container
		main_container = ttk.Frame(self.root)
		main_container.pack(fill=tk.BOTH, expand=True)
		
		# Title
		title_frame = ttk.Frame(main_container, height=100)
		title_frame.pack(fill=tk.X, padx=20, pady=20)
		
		title_label = tk.Label(title_frame, text="🎭 FACE MASK DETECTION SYSTEM 🎭", 
							   font=("Arial", 48, "bold"), fg="#2c3e50")
		title_label.pack()
		
		subtitle_label = tk.Label(title_frame, text="Real-time Face Mask & Person Detection", 
								  font=("Arial", 14), fg="#7f8c8d")
		subtitle_label.pack()
		
		# Video frame
		self.video_frame = ttk.Frame(main_container, relief=tk.SUNKEN)
		self.video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
		
		self.video_label = tk.Label(self.video_frame, bg="black")
		self.video_label.pack(fill=tk.BOTH, expand=True)
		
		# Button frame
		button_frame = ttk.Frame(main_container)
		button_frame.pack(fill=tk.X, padx=20, pady=20)
		
		# Start button
		self.start_btn = tk.Button(button_frame, text="▶ START", command=self.start_detection,
								   font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
								   padx=30, pady=15, relief=tk.RAISED, bd=2, cursor="hand2")
		self.start_btn.pack(side=tk.LEFT, padx=10)
		
		# Stop button
		self.stop_btn = tk.Button(button_frame, text="⏹ STOP", command=self.stop_detection,
								  font=("Arial", 14, "bold"), bg="#e74c3c", fg="white",
								  padx=30, pady=15, relief=tk.RAISED, bd=2, cursor="hand2", state=tk.DISABLED)
		self.stop_btn.pack(side=tk.LEFT, padx=10)
		
		# Status label
		self.status_label = tk.Label(button_frame, text="Status: Ready", 
									  font=("Arial", 12), fg="#27ae60")
		self.status_label.pack(side=tk.LEFT, padx=20)
		
		# Stats label
		self.stats_label = tk.Label(button_frame, text="Faces: 0 | Known: 0", 
									 font=("Arial", 11), fg="#3498db")
		self.stats_label.pack(side=tk.LEFT, padx=20)
		
		# Bind close event
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		
	def start_detection(self):
		if self.is_running:
			return
		
		print("[INFO] Starting detection...")
		self.vs = open_camera()
		if self.vs is None:
			self.status_label.config(text="Status: Camera Error", fg="#e74c3c")
			return
		
		self.is_running = True
		self.start_btn.config(state=tk.DISABLED)
		self.stop_btn.config(state=tk.NORMAL)
		self.status_label.config(text="Status: Running", fg="#27ae60")
		
		# Start detection thread
		self.thread = threading.Thread(target=self.detection_loop, daemon=True)
		self.thread.start()
		
	def stop_detection(self):
		print("[INFO] Stopping detection...")
		self.is_running = False
		self.start_btn.config(state=tk.NORMAL)
		self.stop_btn.config(state=tk.DISABLED)
		self.status_label.config(text="Status: Stopped", fg="#e67e22")
		
		if self.vs is not None:
			self.vs.release()
			self.vs = None
		
		# Clear video display
		blank_image = Image.new('RGB', (640, 480), color='black')
		photo = ImageTk.PhotoImage(blank_image)
		self.video_label.config(image=photo)
		self.video_label.image = photo
		
	def detection_loop(self):
		while self.is_running:
			ok, frame = self.vs.read()
			if not ok or frame is None:
				print("[WARN] Could not read from camera.")
				self.is_running = False
				break
			
			# Resize frame for display
			frame = imutils.resize(frame, width=640)
			
			# Detect faces and masks
			(locs, preds) = detect_and_predict_mask(frame, self.faceNet, self.maskNet)
			
			# Track statistics
			total_faces = len(locs)
			known_faces = 0
			
			# Draw predictions
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				
				# Determine mask status
				mask_label = "Mask" if mask > withoutMask else "No Mask"
				mask_color = (0, 255, 0) if mask_label == "Mask" else (0, 0, 255)
				
				# Recognize face
				face = frame[startY:endY, startX:endX]
				person_name, confidence = recognize_face(face, self.faceRecognizer, self.faceLabelNames)
				
				# Create comprehensive label
				if person_name == "Unknown":
					# Unknown person
					label = "🔍 Unknown - {}".format(mask_label)
					box_color = (0, 165, 255)  # Orange for unknown
				else:
					# Known person
					label = "✓ {} - {}".format(person_name, mask_label)
					box_color = (0, 255, 0)  # Green for known
					known_faces += 1
				
				# Add probabilities
				label = "{}: {:.1f}%".format(label, max(mask, withoutMask) * 100)
				
				# Draw on frame
				cv2.putText(frame, label, (startX, startY - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
				cv2.putText(frame, "Face Conf: {:.1f}%".format(100 - confidence), 
					(startX, startY - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1)
				cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
			
			# Update stats
			self.stats_label.config(text="Faces: {} | Known: {}".format(total_faces, known_faces))
			
			# Convert frame to PhotoImage
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image_pil = Image.fromarray(frame_rgb)
			photo = ImageTk.PhotoImage(image_pil)
			
			# Update label
			self.video_label.config(image=photo)
			self.video_label.image = photo
			
			self.root.update()
	
	def on_closing(self):
		print("[INFO] Closing application...")
		self.is_running = False
		if self.vs is not None:
			self.vs.release()
		self.root.destroy()

# Create and run GUI
if __name__ == "__main__":
	root = tk.Tk()
	app = FaceMaskDetectorGUI(root)
	root.mainloop()
