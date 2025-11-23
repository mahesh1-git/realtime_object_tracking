# realtime_tracking.py
# Updated: combines TensorFlow detection (COCO) + MediaPipe Hands + pen heuristic
# Requirements: opencv-python, tensorflow (or tensorflow-cpu), numpy, mediapipe
# Put a TF2 SavedModel in folder "saved_model" next to this script.

import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp

# Small COCO label map (common classes). COCO class 77 is 'cell phone'.
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "pen",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

# ---------- Centroid Tracker (unchanged, robust) ----------
class CentroidTracker:
    def __init__(self, max_lost=30, max_distance=50):
        self.next_object_id = 0
        self.objects = dict()      # object_id -> centroid (x,y)
        self.lost = dict()         # object_id -> consecutive frames lost
        self.max_lost = max_lost
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.lost[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.lost:
            del self.lost[object_id]

    def update(self, detections):
        # detections: list of (startX,startY,endX,endY)
        input_centroids = []
        for (startX, startY, endX, endY) in detections:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        if len(input_centroids) == 0:
            for object_id in list(self.objects.keys()):
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.deregister(object_id)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            object_id = object_ids[r]
            self.objects[object_id] = input_centroids[c]
            self.lost[object_id] = 0
            assigned_rows.add(r)
            assigned_cols.add(c)

        for i in range(len(input_centroids)):
            if i not in assigned_cols:
                self.register(input_centroids[i])

        for i in range(len(object_centroids)):
            if i not in assigned_rows:
                object_id = object_ids[i]
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.deregister(object_id)

        return self.objects

# -------- TF model helpers ----------
def load_model(path="saved_model"):
    print("Loading TF model from:", path)
    detect_fn = tf.saved_model.load(path)
    print("Model loaded.")
    return detect_fn

def run_inference(detect_fn, frame_rgb):
    # frame_rgb: H,W,3 uint8
    input_tensor = tf.convert_to_tensor(frame_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    results = detect_fn(input_tensor)

    boxes = results.get('detection_boxes')
    scores = results.get('detection_scores')
    classes = results.get('detection_classes')
    num = results.get('num_detections')

    if boxes is None:
        return [], [], []

    boxes = boxes[0].numpy()
    scores = scores[0].numpy()
    classes = classes[0].numpy().astype(np.int32)
    num = int(num[0].numpy()) if num is not None else boxes.shape[0]

    h, w, _ = frame_rgb.shape
    pixel_boxes = []
    for i in range(num):
        ymin, xmin, ymax, xmax = boxes[i]
        startX = int(xmin * w); startY = int(ymin * h)
        endX = int(xmax * w); endY = int(ymax * h)
        pixel_boxes.append((startX, startY, endX, endY))

    return pixel_boxes, scores, classes

# ---------- MediaPipe hands setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Pen heuristic ----------
def detect_possible_pens(hand_bbox, frame_gray, orig_frame):
    """
    hand_bbox: (x1,y1,x2,y2) in pixel coords
    frame_gray: entire frame grayscale image
    orig_frame: original BGR image for drawing
    Returns list of pen-like boxes [(x1,y1,x2,y2), ...]
    Heuristic: look for thin elongated contours within an expanded region around the hand.
    """
    x1, y1, x2, y2 = hand_bbox
    h, w = frame_gray.shape
    pad = 50
    sx = max(0, x1 - pad); sy = max(0, y1 - pad)
    ex = min(w-1, x2 + pad); ey = min(h-1, y2 + pad)
    crop = frame_gray[sy:ey, sx:ex]
    if crop.size == 0:
        return []

    # stronger edges for thin object detection
    edges = cv2.Canny(crop, 50, 150)
    # dilation to connect segments slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 8 and ch > 10:  # very thin but some height
            ar = ch / (cw + 1e-6)
            if ar > 2.5 and cw * ch > 20:  # elongated and not a speck
                # map back to original coords
                bx1 = sx + x; by1 = sy + y; bx2 = bx1 + cw; by2 = by1 + ch
                pen_boxes.append((bx1, by1, bx2, by2))
    return pen_boxes

# -------- Main ----------
def main():
    detect_fn = load_model("saved_model")
    tracker = CentroidTracker(max_lost=15, max_distance=80)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    hands_detector = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=4,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    fps = 0.0
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # small resize for speed if frame is large (optional)
            # frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- TF detection (COCO objects) ---
            try:
                boxes, scores, classes = run_inference(detect_fn, frame_rgb)
            except Exception as e:
                print("Inference error:", e)
                boxes, scores, classes = [], [], []

            # Filter by score
            threshold = 0.45
            filtered_boxes = []
            filtered_scores = []
            filtered_classes = []
            for (box, score, cls) in zip(boxes, scores, classes):
                if score >= threshold:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_classes.append(int(cls))

            # Update tracker using bounding boxes (all detections)
            objects = tracker.update(filtered_boxes)

            # Draw TF boxes + labels
            for (box, score, cls) in zip(filtered_boxes, filtered_scores, filtered_classes):
                sx, sy, ex, ey = box
                label = COCO_LABELS.get(cls, f"cls{cls}")
                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 200, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (sx, max(15, sy-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # Draw tracker centroids & IDs
            for oid, centroid in objects.items():
                cx, cy = centroid
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"Obj{oid}", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # --- MediaPipe Hands detection ---
            hand_results = hands_detector.process(frame_rgb)
            pen_boxes_all = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # draw landmarks on original frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # compute bounding box around landmarks
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    min_x = int(min(xs) * w); max_x = int(max(xs) * w)
                    min_y = int(min(ys) * h); max_y = int(max(ys) * h)
                    # expand slightly
                    pad = 10
                    bx1 = max(0, min_x - pad); by1 = max(0, min_y - pad)
                    bx2 = min(w-1, max_x + pad); by2 = min(h-1, max_y + pad)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(frame, "Hand", (bx1, max(15, by1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

                    # run pen heuristic inside region
                    pens = detect_possible_pens((bx1, by1, bx2, by2), frame_gray, frame)
                    for (px1, py1, px2, py2) in pens:
                        pen_boxes_all.append((px1, py1, px2, py2))
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
                        cv2.putText(frame, "Possible Pen", (px1, max(15, py1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            # If TF detects a cell phone, label is already drawn. We additionally highlight if phone near hand
            # draw small indicator if phone is close to any hand bbox
            # (optional) compute proximity:
            for i, box in enumerate(filtered_boxes):
                sx, sy, ex, ey = box
                # check distance to any detected hand boxes
                for hb in (pen_boxes_all if len(pen_boxes_all)>0 else []):
                    # if pen heuristic box overlaps phone box, highlight
                    px1, py1, px2, py2 = hb
                    inter_x1 = max(sx, px1); inter_y1 = max(sy, py1)
                    inter_x2 = min(ex, px2); inter_y2 = min(ey, py2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        cv2.putText(frame, "Phone near hand", (sx, ey+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)

            # overlay people count and general counts
            people_count = sum(1 for c in filtered_classes if c == 1)
            cv2.putText(frame, f"People: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            # optionally show total TF detections
            cv2.putText(frame, f"Detections: {len(filtered_boxes)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)

            # fps
            frame_count += 1
            if frame_count % 10 == 0:
                t1 = time.time()
                fps = 10.0 / (t1 - t0)
                t0 = t1
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Detections", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker = CentroidTracker(max_lost=15, max_distance=80)
                print("Tracker reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()

if __name__ == "__main__":
    main()
