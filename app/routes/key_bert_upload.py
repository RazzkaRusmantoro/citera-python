from flask import Blueprint, request, jsonify
from keybert import KeyBERT
import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX
import tempfile
import os

# Initialize KeyBERT model
kw_model = KeyBERT('paraphrase-MiniLM-L3-v2')

keyword_extraction_bp = Blueprint('keyword_extraction', __name__)

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text() + "\n\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

@keyword_extraction_bp.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    try:
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file.filename.lower().endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        
        # Extract keywords using KeyBERT
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=1  # Adjust number of keywords as needed
        )
        
        # Format the response
        result = {
            "keywords": [{"keyword": kw[0], "score": float(kw[1])} for kw in keywords],
            "text_length": len(text)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary files
        try:
            os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass