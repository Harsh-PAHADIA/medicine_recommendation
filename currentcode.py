import os
import random
import numpy as np
import pandas as pd
import cv2
import pytesseract
import google.generativeai as genai
from flask import Flask, render_template, request
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Gemini API
genai.configure(api_key="AIzaSyATOpvcuT_PuKgH9seqzunxkOKswLVNv0g")  # Replace with your actual API key

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load dataset
df = pd.read_csv("static/Training.csv")
X = df.drop(columns=["prognosis"])
y = df["prognosis"]
le = LabelEncoder()
Y = le.fit_transform(y)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
models = {
    "SVC": SVC(kernel='linear'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB": MultinomialNB()
}
for model in models.values():
    model.fit(X_train, y_train)

# Disease prediction function
def predict_disease(user_symptoms):
    symptom_list = list(X.columns)
    user_input_vector = np.zeros(len(symptom_list))
    for symptom in user_symptoms:
        if symptom in symptom_list:
            user_input_vector[symptom_list.index(symptom)] = 1
    user_input_vector = user_input_vector.reshape(1, -1)
    predictions = [model.predict(user_input_vector)[0] for model in models.values()]
    prediction_counts = {disease: predictions.count(disease) for disease in set(predictions)}
    predicted_disease = max(prediction_counts, key=prediction_counts.get)
    return le.inverse_transform([predicted_disease])[0]

# Extract text from images
def extract_text_from_image(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

# Analyze medical report using Gemini AI
def analyze_medical_report(text, disease, user_symptoms):
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    The patient has reported symptoms: {', '.join(user_symptoms)}.
    A machine learning model predicted the disease: {disease}.
    Below is the extracted text from the medical report:
    {text}

    Based on this information, please provide:
    1. Does the report contradict the predicted disease? (Yes/No + explanation)
    2. The most likely diagnosis based on report findings.
    3. A concise summary of the patient's condition.
    4. Recommended medications (mention specific medicine names if applicable).
    """

    response = model.generate_content(prompt).text
    sections = response.split("\n\n")

    return {
        "contradiction": sections[0] if len(sections) > 0 else "No data",
        "likely_diagnosis": sections[1] if len(sections) > 1 else "No data",
        "summary": sections[2] if len(sections) > 2 else "No data",
        "medications": sections[3] if len(sections) > 3 else "No medications prescribed",
    }

@app.route('/')
def upload_page():
    symptoms_list = list(X.columns)
    return render_template('upload.html', symptoms_list=symptoms_list)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    symptoms = request.form.getlist('symptoms')
    full_text = ""

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            if file.filename.lower().endswith(".pdf"):
                images = convert_from_path(file_path)
                for i, image in enumerate(images):
                    temp_path = os.path.join(UPLOAD_FOLDER, f"page_{i}.png")
                    image.save(temp_path, "PNG")
                    full_text += extract_text_from_image(temp_path) + "\n"
            else:
                full_text += extract_text_from_image(file_path) + "\n"

    disease = predict_disease(symptoms) if symptoms else "Unknown"
    analysis = analyze_medical_report(full_text, disease, symptoms) if full_text.strip() else {}

    return render_template('index.html', analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True)
