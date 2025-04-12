import os
import random
import numpy as np
import pandas as pd
import time
import pytesseract
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from pdf2image import convert_from_path
from PIL import Image
import openai

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = openai.OpenAI(
    base_url=" ",
    api_key=" "
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load training data for disease prediction model
file_path = "static/Training.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["prognosis"])
y = df["prognosis"]
le = LabelEncoder()
Y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialize machine learning models
models = {
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB": MultinomialNB()
}

for model in models.values():
    model.fit(X_train, y_train)

# Predict disease based on user symptoms
def predict_disease(user_symptoms):
    symptom_list = sorted(list(X.columns))
    user_input_dict = {symptom: 0 for symptom in X.columns}
    
    for symptom in user_symptoms:
        if symptom in user_input_dict:
            user_input_dict[symptom] = 1
    
    user_input_df = pd.DataFrame([user_input_dict])
    
    knn_probs = models["KNeighbors"].predict_proba(user_input_df)[0]
    nb_probs = models["MultinomialNB"].predict_proba(user_input_df)[0]
    
    knn_top_3 = np.argsort(knn_probs)[-3:][::-1]
    nb_top_3 = np.argsort(nb_probs)[-3:][::-1]
    
    knn_diseases = le.inverse_transform(knn_top_3)
    nb_diseases = le.inverse_transform(nb_top_3)
    
    likely_diseases = list(dict.fromkeys(list(knn_diseases) + list(nb_diseases)))
    return likely_diseases


# Extract text from medical reports
def extract_text_from_file(file_path):
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path)
        text = "\n".join([pytesseract.image_to_string(image) for image in images])
    else:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    return text

# Analyze medical report using OpenAI GPT model
def analyze_medical_report(text, likely_diseases, user_symptoms):
    time.sleep(1)
    diseases_str = ", ".join(likely_diseases)
    analysis_prompt = f"""Patient's symptoms suggest these possible diseases: {diseases_str}.
    
Act as a medical expert analyzing a patient case. Consider these elements:

Patient Symptoms: {user_symptoms}
Most Likely Diseases: {diseases_str}
Medical Report Content:
{text}

Provide analysis in this exact structure:
1. Disease Confirmation: Confirm if report supports predictions (Yes/No/Partial) with reasoning.
2. Most Likely Diagnosis: Provide the most probable diagnosis based on all data.
3. Condition Summary: 2-3 sentence overview of patient's health status.
4. Recommended Medications: Bullet list of medicines with generic names.
5. Pre-Appointment Prep: List of 5 actionable steps before a doctor visit.
6. Home Care Advice: Practical home treatments with safety cautions.

Use clear medical terminology but avoid markdown formatting. Maintain strict numbering."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a medical analysis AI. Provide structured and factual responses."},
            {"role": "user", "content": analysis_prompt}
        ],
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9
    )
    
    sections = response.choices[0].message.content.split("\n\n")
    while len(sections) < 6:
        sections.append("Unavailable")
    
    analysis_result = {
        "confirmation": sections[0].split(":", 1)[1].strip() if ":" in sections[0] else "Unavailable",
        "diagnosis": sections[1].split(":", 1)[1].strip() if ":" in sections[1] else "Unavailable",
        "summary": sections[2].split(":", 1)[1].strip() if ":" in sections[2] else "Unavailable",
        "medications": sections[3].split(":", 1)[1].strip() if ":" in sections[3] else "None recommended",
        "pre_appointment": sections[4].split(":", 1)[1].strip() if ":" in sections[4] else "Unavailable",
        "home_care": sections[5].split(":", 1)[1].strip() if ":" in sections[5] else "Unavailable"
    }
    
    return analysis_result

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
            extracted_text = extract_text_from_file(file_path)
            full_text += extracted_text + "\n"
    
    likely_diseases = predict_disease(symptoms) if symptoms else ["Unknown"]
    analysis = analyze_medical_report(full_text, likely_diseases, symptoms) if full_text.strip() else {}
    
    return render_template('index.html', analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True)
