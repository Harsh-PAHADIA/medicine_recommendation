from flask import Flask, render_template, request
import cv2
import pytesseract
import google.generativeai as genai
import os
import numpy as np
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/preprocessed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure Tesseract OCR (Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Gemini API key
GOOGLE_API_KEY = "AIzaSyATOpvcuT_PuKgH9seqzunxkOKswLVNv0g"
genai.configure(api_key=GOOGLE_API_KEY)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, save_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(save_path, processed)
    return processed

def extract_text_from_image(image_path, save_path):
    processed_img = preprocess_image(image_path, save_path)
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(processed_img, config=custom_config, lang="eng")
    return extracted_text.strip()

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for i, image in enumerate(images):
        image_path = os.path.join(PROCESSED_FOLDER, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        extracted_text += f"\n--- Page {i + 1} ---\n" + extract_text_from_image(image_path, image_path)
    return extracted_text

def analyze_medical_report(text):
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    Analyze the following medical report and provide these conclusions:
    
    1. **Medicine Recommended** (if applicable)
    2. **Disease Description**
    3. **Suggested Diet (Foods to Eat & Avoid)**
    4. **Precautions Before Visiting a Doctor**
    
    If the report contains incomplete information, infer the most relevant diagnosis.
    
    **Medical Report Extracted Text:**
    ```{text}```
    """
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    full_text = ""
    image_paths = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit(".", 1)[1].lower()
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            if file_ext in ["jpg", "jpeg", "png"]:
                processed_path = os.path.join(PROCESSED_FOLDER, filename)
                extracted_text = extract_text_from_image(file_path, processed_path)
                image_paths.append(file_path)
            elif file_ext == "pdf":
                extracted_text = extract_text_from_pdf(file_path)
            else:
                continue  # Skip unsupported file types

            full_text += extracted_text + "\n"

    analysis = analyze_medical_report(full_text) if full_text.strip() else "No text detected. Try clearer images."

    return render_template('index.html', analysis=analysis, image_paths=image_paths)

if __name__ == '__main__':
    app.run(debug=True)
