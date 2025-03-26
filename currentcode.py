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
PROCESSED_FOLDER = "static/preprocessed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

genai.configure(api_key="AIzaSyATOpvcuT_PuKgH9seqzunxkOKswLVNv0g")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

file_path = "static/Training.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["prognosis"])
y = df["prognosis"]
le = LabelEncoder()
Y = le.fit_transform(y)
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

def predict_disease(user_symptoms):
    symptom_list = list(X.columns)
    user_input_vector = np.zeros(len(symptom_list))
    for symptom in user_symptoms:
        if symptom in symptom_list:
            user_input_vector[symptom_list.index(symptom)] = 1
    user_input_vector = user_input_vector.reshape(1, -1)
    predictions = [model.predict(user_input_vector)[0] for model in models.values()]
    prediction_counts = {disease: predictions.count(disease) for disease in set(predictions)}
    max_count = max(prediction_counts.values())
    top_diseases = [d for d, count in prediction_counts.items() if count == max_count]
    predicted_disease = random.choice(top_diseases)
    return le.inverse_transform([predicted_disease])[0]

def analyze_medical_report(text, disease, user_symptoms):
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    The user has the following symptoms: {user_symptoms}, and an ML model predicted the disease as {disease}.
    The extracted medical report is:
    {text}
    Please provide:
    1. Whether the report contradicts the predicted disease.
    2. The most likely diagnosis.
    3. A brief summary of the patient's condition.
    4. List of names of recommended medicines.
    """
    response = model.generate_content(prompt).text
    sections = response.split("\n\n")
    return {
        "contradiction": sections[0] if len(sections) > 0 else "No data",
        "likely_diagnosis": sections[1] if len(sections) > 1 else "No data",
        "summary": sections[2] if len(sections) > 2 else "No data",
        "medications": sections[3] if len(sections) > 3 else "No data",
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
            extracted_text = "Sample extracted text"  # Placeholder for OCR processing
            full_text += extracted_text + "\n"
    disease = predict_disease(symptoms) if symptoms else "Unknown"
    analysis = analyze_medical_report(full_text, disease, symptoms) if full_text.strip() else {}
    return render_template('index.html', analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True)
