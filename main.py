from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import base64
import mediapipe as mp
import math
import tensorflow as tf
from sklearn.cluster import KMeans

app = FastAPI()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Golden Ratio
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

# Neural network model for beauty score prediction
def build_beauty_score_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

beauty_model = build_beauty_score_model()

# Train the model (you should replace this with actual training data)
def train_model(model):
    X = np.array([[1.62, 0.98, 0.8], [1.5, 1.1, 0.9], [1.55, 0.95, 0.88]])
    y = np.array([0.9, 0.7, 0.8])
    model.fit(X, y, epochs=100)

train_model(beauty_model)

# Helper functions (reuse the existing functions from the Flask implementation)
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def categorize_beauty_score(score):
    if score >= 0.85:
        return "Perfect (0.85)", "green"
    elif score >= 0.7:
        return "Good (0.7)", "blue"
    elif score >= 0.5:
        return "Average (0.5)", "orange"
    else:
        return "Bad", "red"

def get_skin_regions(image, landmarks):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    regions = [
        [10, 338, 297, 332],  # Forehead
        [454, 423, 426, 411],  # Left cheek
        [234, 127, 162, 21],   # Right cheek
        [2, 326, 328, 330]     # Nose
    ]

    for region in regions:
        pts = np.array([(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in region], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

    return cv2.bitwise_and(image, image, mask=mask)

def classify_skin_tone(rgb_color):
    skin_tones = {
        "Very Light": [255, 224, 196],
        "Light": [234, 192, 134],
        "Medium Light": [255, 173, 96],
        "Medium": [191, 137, 84],
        "Medium Dark": [141, 85, 36],
        "Dark": [89, 47, 22],
        "Very Dark": [62, 30, 10]
    }

    distances = {tone: np.linalg.norm(np.array(rgb_color) - np.array(color)) for tone, color in skin_tones.items()}
    return min(distances, key=distances.get)

def analyze_skin_tone(image, landmarks):
    skin_regions = get_skin_regions(image, landmarks)
    rgb_skin = cv2.cvtColor(skin_regions, cv2.COLOR_BGR2RGB)
    pixels = rgb_skin[rgb_skin[:,:,0] != 0].reshape(-1, 3)
    
    if len(pixels) == 0:
        return "Unable to detect skin", "#FFFFFF"
    
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    skin_tone = classify_skin_tone(dominant_color)
    color_hex = '#{:02x}{:02x}{:02x}'.format(*dominant_color)
    
    return skin_tone, color_hex

def calculate_beauty_ratios(landmarks, model):
    face_height = calculate_distance(landmarks[10], landmarks[152])
    face_width = calculate_distance(landmarks[234], landmarks[454])
    eye_width = calculate_distance(landmarks[33], landmarks[263])
    nose_width = calculate_distance(landmarks[1], landmarks[5])

    face_ratio = face_height / face_width if face_width != 0 else 0
    eye_to_face_ratio = eye_width / face_width if face_width != 0 else 0
    nose_to_face_ratio = nose_width / face_width if face_width != 0 else 0

    ratios = np.array([[face_ratio, eye_to_face_ratio, nose_to_face_ratio]])
    score = model.predict(ratios)

    return score[0][0], face_ratio, eye_to_face_ratio, nose_to_face_ratio

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return JSONResponse(status_code=400, content={'error': 'No face detected in the image'})

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(face_landmarks.landmark[i].x * image.shape[1],
                  face_landmarks.landmark[i].y * image.shape[0],
                  face_landmarks.landmark[i].z * image.shape[1]) for i in range(468)]

    score, face_ratio, eye_ratio, nose_ratio = calculate_beauty_ratios(landmarks, beauty_model)
    score_category, color = categorize_beauty_score(score)
    skin_tone, skin_color = analyze_skin_tone(image, face_landmarks.landmark)

    suggestions = []
    if abs(face_ratio - GOLDEN_RATIO) > 0.2:
        suggestions.append("Adjust face proportions for better symmetry")
    if abs(eye_ratio - GOLDEN_RATIO) > 0.2:
        suggestions.append("Eye proportions could be improved")
    if abs(nose_ratio - GOLDEN_RATIO) > 0.2:
        suggestions.append("Consider adjusting nose proportions")

    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'beauty_score': float(score),
        'score_category': score_category,
        'category_color': color,
        'face_ratio': float(face_ratio),
        'eye_ratio': float(eye_ratio),
        'nose_ratio': float(nose_ratio),
        'skin_tone': skin_tone,
        'skin_color': skin_color,
        'suggestions': suggestions,
        'image': f'data:image/jpeg;base64,{image_base64}'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)