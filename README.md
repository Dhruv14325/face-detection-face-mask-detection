# Face Mask Detection System

A complete face mask detection system with person recognition using deep learning. Detects whether people are wearing masks and recognizes known individuals.

## Features

✅ **Real-time Face Mask Detection** - Uses MobileNetV2 with 95%+ accuracy  
✅ **Face Recognition** - Identifies known individuals  
✅ **GUI Interface** - User-friendly fullscreen application  
✅ **Easy Face Capture** - Simple tool to add known faces via camera  
✅ **Statistics Display** - Shows detection metrics in real-time  

## Project Structure

```
Face-Mask-Detection-master/
├── app.py                           # Main GUI application
├── capture_known_faces.py           # Tool to capture known faces from camera
├── train_mask_detector.py           # Training script for mask detector
├── mask_detector.model              # Pre-trained mask detection model
├── requirements.txt                 # Python dependencies
├── dataset/
│   ├── with_mask/                   # Training images with masks
│   └── without_mask/                # Training images without masks
├── face_detector/                   # Face detection model files
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── known_faces/                     # Reference photos for known individuals
    ├── README.md
    ├── Alice/
    │   ├── alice_1.jpg
    │   └── alice_2.jpg
    └── Bob/
        ├── bob_1.jpg
        └── bob_2.jpg
```

## Installation

### 1. Clone/Download the project
```bash
cd Face-Mask-Detection-master
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Main Detection App

```bash
python app.py
```

**GUI Features:**
- 🎭 **Title** - Large, clear application title
- **Live Video Feed** - Real-time camera detection
- **▶ START Button** - Begin detection
- **⏹ STOP Button** - Stop detection
- **Status Indicator** - Shows running/stopped status
- **Statistics** - Displays total faces and known faces count

**Detection Labels:**
- **✓ Name - Mask** - Green box (known person with mask)
- **✓ Name - No Mask** - Green box (known person without mask)
- **🔍 Unknown - Mask** - Orange box (unknown person with mask)
- **🔍 Unknown - No Mask** - Orange box (unknown person without mask)

### Option 2: Capture Known Faces

Before using face recognition, add reference photos:

```bash
python capture_known_faces.py
```

**Steps:**
1. Enter a person's name (e.g., "Alice")
2. Click **▶ START CAMERA**
3. Position your face in the green circle
4. Click **📷 CAPTURE** (repeat 3-5 times for best results)
5. Click **⏹ STOP** when done
6. Photos automatically saved to `known_faces/YourName/`

**Tips:**
- ✅ Use clear, well-lit photos
- ✅ Capture from slightly different angles
- ✅ Include neutral and smiling expressions
- ✅ Add 3-5 photos per person minimum

### Option 3: Train Custom Mask Detector

If you have a custom dataset:

```bash
python train_mask_detector.py
```

This trains the mask detection model on your dataset in `dataset/with_mask/` and `dataset/without_mask/`.

## How It Works

### 1. **Face Detection**
- Uses SSD (Single Shot MultiBox Detector)
- Pre-trained on WIDER Face dataset
- Detects faces in real-time

### 2. **Mask Detection**
- Uses MobileNetV2 neural network
- Binary classification: "Mask" or "No Mask"
- Fast inference for real-time performance

### 3. **Face Recognition**
- Uses LBPH (Local Binary Patterns Histograms)
- Trained on known faces from `known_faces/` directory
- Automatically retrains when new faces are added

## Customization

### Adjust Detection Confidence

Edit `app.py` to change the minimum confidence threshold:

```python
RECOGNITION_THRESHOLD = 75.0  # Increase for stricter matching
```

### Change Camera Index

If you have multiple cameras:

```python
CAMERA_INDEXES = (0, 1, 2, 3)  # Will try cameras in order
```

### Adjust Model Parameters

In `train_mask_detector.py`:

```python
INIT_LR = 1e-4          # Learning rate
EPOCHS = 20             # Training epochs
BS = 32                 # Batch size
```

## Troubleshooting

### Camera Not Working
- Ensure camera permissions are granted
- Try different camera indices in the app
- Check if camera is being used by another application

### Face Recognition Not Working
1. Add more photos to `known_faces/` directory
2. Delete `face_recognizer.yml` and `face_labels.pickle`
3. Restart the app to retrain

### Low Detection Accuracy
- Ensure good lighting
- Add more training images for custom training
- Adjust confidence thresholds

### Performance Issues
- Reduce frame resolution
- Lower epoch count for training
- Use a GPU if available (modify TensorFlow config)

## System Requirements

- **Python**: 3.7+
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional (CPU will work but slower)
- **Camera**: Webcam or USB camera

## Dependencies

- TensorFlow 2.10.1
- OpenCV 4.10.0
- NumPy 1.23.5
- Pillow 9.5.0
- scikit-learn 1.3.2
- Imutils 0.5.4

## Performance Notes

- **Mask Detection**: ~30-40 FPS on CPU
- **Face Recognition**: ~50-60 FPS on CPU
- GPU usage would improve performance 5-10x

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or improvements, check the project files or modify parameters as needed.
