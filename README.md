# realtime_object_tracking
Real-time object tracking using TensorFlow and OpenCV

This project demonstrates real-time object detection and tracking using a combination of TensorFlow, OpenCV, and MediaPipe.
It captures live video from a webcam, detects multiple objects, identifies hands, and tracks movement across frames using a centroid-based tracking algorithm.

The system is designed to be simple, fast, and easy to understand — suitable for academic projects, mini-project submissions, and real-world applications.

📌 Project Overview

This project performs:

✔ Real-Time Object Detection

Uses a TensorFlow SSD MobileNet model to detect common objects from the COCO dataset.

Recognizes objects such as:

Person

Cell phone

Laptop

Bottle

Chair

Many more (COCO classes)

✔ Hand Detection Using MediaPipe

Detects both left and right hands accurately.

Draws landmarks on the hand (fingers, joints).

Creates bounding boxes for each detected hand.

✔ Object Tracking

Each detected object is assigned a unique ID like Obj0, Obj1, etc.

IDs remain consistent across frames as long as the object stays visible.

Uses a centroid tracker to maintain stable tracking.

✔ Pen-Like Object Detection (Heuristic)

A small image-processing method attempts to detect thin, elongated objects (possible pens) near detected hands.

Highlights “Possible Pen” regions based on contour analysis.

✔ People Counting

Automatically counts the number of people visible in the frame.

Updates count in real time.

✔ FPS Display (Speed Monitor)

Shows live processing speed in Frames Per Second.

Helps evaluate performance of the system.

🧠 How It Works (Simple Explanation)

The webcam sends video frames to the program.

Each frame is processed by:

TensorFlow model → detects objects

MediaPipe → detects hands

Detected bounding boxes are passed to a tracking algorithm.

The tracker assigns IDs so each object is tracked as it moves.

The system draws:

Boxes around objects

Labels showing class and confidence

IDs above tracked objects

Hand markers and “Possible Pen” areas

Final processed frame is displayed live on screen.

🔧 Technologies Used
Technology	Purpose
Python	Core programming language
OpenCV	Video capture, drawing, frame processing
TensorFlow	Deep learning model for object detection
MediaPipe Hands	High-accuracy hand tracking
NumPy	Numerical operations
Centroid Tracking Algorithm	Identifies and tracks moving objects
📂 Project Structure
realtime_object_tracking/
│
├── saved_model/                 # TensorFlow model files
├── realtime_tracking.py         # Main application
├── .venv/                       # Virtual environment
└── README.md                    # Project documentation

▶️ How to Run

Open project folder in VS Code

Activate the virtual environment:

.\.venv\Scripts\activate


Run the project:

python realtime_tracking.py


Controls:

Press q to quit

Press r to reset tracker IDs

⭐ Key Features

Multi-object detection

Stable object tracking

Hand landmark detection

People counting

Pen-like object analysis

High-speed real-time performance

Simple and easy-to-understand implementation

📘 Use Cases

Mini projects for college

Real-time monitoring systems

Gesture recognition foundation

Smart classroom monitoring

Industrial tool/worker tracking

Beginner-friendly AI/ML project

🔮 Possible Improvements

Add DeepSORT for advanced tracking

Add face detection or face recognition

Train custom models for pen/bottle/weapon detection

Add region-based counting (entry/exit detection)

Improve GUI for user interaction

🎉 Conclusion

This project provides a strong foundation for anyone beginning with computer vision, AI, or real-time processing.
It combines deep learning, tracking algorithms, and hand detection in a simple, well-structured way — making it ideal for practical demonstrations and academic presentations.
