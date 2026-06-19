# import the necessary packages
import cv2
import os
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading

class CaptureKnownFacesApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Capture Known Faces")
		self.root.geometry("900x700")
		
		# Variables
		self.camera = None
		self.is_capturing = False
		self.person_name = ""
		self.capture_count = 0
		self.known_faces_dir = "known_faces"
		
		# Ensure known_faces directory exists
		if not os.path.exists(self.known_faces_dir):
			os.makedirs(self.known_faces_dir)
		
		self.create_gui()
		
	def create_gui(self):
		# Title
		title_label = tk.Label(self.root, text="📸 Capture Known Faces", 
							   font=("Arial", 36, "bold"), fg="#2c3e50")
		title_label.pack(pady=20)
		
		subtitle_label = tk.Label(self.root, text="Add reference photos for face recognition", 
								  font=("Arial", 12), fg="#7f8c8d")
		subtitle_label.pack()
		
		# Video frame
		video_container = ttk.Frame(self.root, relief=tk.SUNKEN)
		video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
		
		self.video_label = tk.Label(video_container, bg="black")
		self.video_label.pack(fill=tk.BOTH, expand=True)
		
		# Input frame
		input_frame = ttk.Frame(self.root)
		input_frame.pack(fill=tk.X, padx=20, pady=10)
		
		input_label = tk.Label(input_frame, text="Person Name:", font=("Arial", 11))
		input_label.pack(side=tk.LEFT, padx=5)
		
		self.name_entry = tk.Entry(input_frame, font=("Arial", 11), width=25)
		self.name_entry.pack(side=tk.LEFT, padx=5)
		
		# Button frame
		button_frame = ttk.Frame(self.root)
		button_frame.pack(fill=tk.X, padx=20, pady=15)
		
		# Start button
		self.start_btn = tk.Button(button_frame, text="▶ START CAMERA", command=self.start_camera,
								   font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
								   padx=20, pady=10, relief=tk.RAISED, bd=2, cursor="hand2")
		self.start_btn.pack(side=tk.LEFT, padx=5)
		
		# Capture button
		self.capture_btn = tk.Button(button_frame, text="📷 CAPTURE", command=self.capture_photo,
									  font=("Arial", 12, "bold"), bg="#3498db", fg="white",
									  padx=20, pady=10, relief=tk.RAISED, bd=2, cursor="hand2", state=tk.DISABLED)
		self.capture_btn.pack(side=tk.LEFT, padx=5)
		
		# Stop button
		self.stop_btn = tk.Button(button_frame, text="⏹ STOP", command=self.stop_camera,
								  font=("Arial", 12, "bold"), bg="#e74c3c", fg="white",
								  padx=20, pady=10, relief=tk.RAISED, bd=2, cursor="hand2", state=tk.DISABLED)
		self.stop_btn.pack(side=tk.LEFT, padx=5)
		
		# Status label
		self.status_label = tk.Label(self.root, text="Status: Ready", 
									  font=("Arial", 11), fg="#27ae60")
		self.status_label.pack(pady=5)
		
		# Info label
		self.info_label = tk.Label(self.root, text="Captured: 0 photos", 
								   font=("Arial", 10), fg="#3498db")
		self.info_label.pack(pady=5)
		
		# Instructions label
		instructions = tk.Label(self.root, 
								text="Instructions: 1) Enter person name  2) Click START CAMERA  3) Position face in frame  4) Click CAPTURE (repeat for multiple photos)  5) Click STOP when done",
								font=("Arial", 9), fg="#7f8c8d", wraplength=800)
		instructions.pack(pady=10)
		
		# Bind close event
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		
	def start_camera(self):
		self.person_name = self.name_entry.get().strip()
		
		if not self.person_name:
			messagebox.showerror("Error", "Please enter a person name!")
			return
		
		if len(self.person_name) < 2:
			messagebox.showerror("Error", "Name must be at least 2 characters long!")
			return
		
		# Create person directory
		person_dir = os.path.join(self.known_faces_dir, self.person_name)
		if not os.path.exists(person_dir):
			os.makedirs(person_dir)
			self.capture_count = 0
		else:
			# Count existing photos
			existing_photos = len([f for f in os.listdir(person_dir) 
									if f.endswith(('.jpg', '.jpeg', '.png'))])
			self.capture_count = existing_photos
		
		print(f"[INFO] Starting camera for {self.person_name}")
		print(f"[INFO] Saving to: {person_dir}")
		
		# Open camera
		self.camera = cv2.VideoCapture(0)
		if not self.camera.isOpened():
			messagebox.showerror("Error", "Could not open camera!")
			return
		
		self.is_capturing = True
		self.start_btn.config(state=tk.DISABLED)
		self.capture_btn.config(state=tk.NORMAL)
		self.stop_btn.config(state=tk.NORMAL)
		self.name_entry.config(state=tk.DISABLED)
		self.status_label.config(text=f"Status: Capturing for {self.person_name}", fg="#27ae60")
		self.update_info()
		
		# Start video update thread
		self.thread = threading.Thread(target=self.video_loop, daemon=True)
		self.thread.start()
		
	def video_loop(self):
		while self.is_capturing and self.camera is not None:
			ret, frame = self.camera.read()
			if not ret:
				break
			
			# Resize for display
			frame = cv2.resize(frame, (640, 480))
			
			# Add instructions on frame
			cv2.putText(frame, f"Capturing: {self.person_name}", (20, 40),
					   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv2.putText(frame, f"Photos: {self.capture_count}", (20, 80),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
			
			# Draw circle in center for face positioning
			h, w = frame.shape[:2]
			cv2.circle(frame, (w//2, h//2), 100, (0, 255, 0), 2)
			cv2.putText(frame, "Position face in circle", (w//2 - 150, h//2 + 150),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			
			# Convert to RGB for display
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image_pil = Image.fromarray(frame_rgb)
			photo = ImageTk.PhotoImage(image_pil)
			
			self.video_label.config(image=photo)
			self.video_label.image = photo
			
			self.root.update()
	
	def capture_photo(self):
		if not self.is_capturing or self.camera is None:
			messagebox.showerror("Error", "Camera not running!")
			return
		
		ret, frame = self.camera.read()
		if not ret:
			messagebox.showerror("Error", "Failed to capture image!")
			return
		
		# Save photo
		person_dir = os.path.join(self.known_faces_dir, self.person_name)
		self.capture_count += 1
		photo_name = f"{self.person_name}_{self.capture_count:03d}.jpg"
		photo_path = os.path.join(person_dir, photo_name)
		
		cv2.imwrite(photo_path, frame)
		
		print(f"[INFO] Captured: {photo_path}")
		self.status_label.config(text=f"✓ Photo saved!", fg="#27ae60")
		self.update_info()
		
		# Reset status after 1 second
		self.root.after(1000, lambda: self.status_label.config(
			text=f"Status: Capturing for {self.person_name}", fg="#27ae60"))
	
	def update_info(self):
		self.info_label.config(text=f"Captured: {self.capture_count} photos")
	
	def stop_camera(self):
		print("[INFO] Stopping camera...")
		self.is_capturing = False
		
		if self.camera is not None:
			self.camera.release()
			self.camera = None
		
		self.start_btn.config(state=tk.NORMAL)
		self.capture_btn.config(state=tk.DISABLED)
		self.stop_btn.config(state=tk.DISABLED)
		self.name_entry.config(state=tk.NORMAL)
		
		self.status_label.config(
			text=f"✓ Saved {self.capture_count} photos for {self.person_name}!", 
			fg="#27ae60")
		
		# Clear video display
		blank_image = Image.new('RGB', (640, 480), color='black')
		photo = ImageTk.PhotoImage(blank_image)
		self.video_label.config(image=photo)
		self.video_label.image = photo
		
		# Clear inputs
		self.name_entry.delete(0, tk.END)
		self.person_name = ""
		self.capture_count = 0
		self.update_info()
		
		messagebox.showinfo("Success", 
						   "Face capture complete! Remember to delete face_recognizer.yml and "
						   "face_labels.pickle to retrain the recognizer.")
	
	def on_closing(self):
		print("[INFO] Closing application...")
		self.is_capturing = False
		if self.camera is not None:
			self.camera.release()
		self.root.destroy()

if __name__ == "__main__":
	root = tk.Tk()
	app = CaptureKnownFacesApp(root)
	root.mainloop()
