from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import spacy
import requests
from openai import OpenAI
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import re

rephrase_bp = Blueprint('rephrase', __name__)

# Initialize models once
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT('paraphrase-MiniLM-L3-v2')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
# Load environment variables
env_path = Path(__file__).resolve().parents[3] / '.env.local'
load_dotenv(dotenv_path=env_path)

OPENAI_KEY = os.environ.get("OPENAI_KEY")
SEMANTIC_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

@rephrase_bp.route('/rephrase', methods=['POST'])
def rephrase():
    try:
        data = request.get_json()
        cited_sentence = data.get('citedSentence', '')
        
        if not cited_sentence:
            return jsonify({'error': 'No sentence provided'}), 400
        
        # Call OpenAI to generate rephrases
        rephrases = generate_rephrases(cited_sentence)
        
        return jsonify({
            'success': True,
            'rephrases': rephrases
        })
        
    except Exception as e:
        print(f"Error in rephrase endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def generate_rephrases(sentence):
    try:
        prompt = f"""
        Please provide 4 different rephrasings of the following academic sentence. 
        Each rephrasing should maintain the original meaning but use different wording and structure.
        Return ONLY a JSON array of the 4 rephrasings, no additional text.
        
        Original sentence: "{sentence}"
        
        Example format: ["Rephrase 1", "Rephrase 2", "Rephrase 3", "Rephrase 4"]
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rephrases academic sentences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract and parse the JSON response
        content = response.choices[0].message.content.strip()
        rephrases = json.loads(content)

        print("These are the rephrases", rephrases)
        
        return rephrases[:4]
        
    except Exception as e:
        print(f"Error generating rephrases: {str(e)}")
        # Fallback: return simple rephrases
        return [
            f"Rephrased version 1: {sentence}",
            f"Rephrased version 2: {sentence}",
            f"Rephrased version 3: {sentence}",
            f"Rephrased version 4: {sentence}"
        ]