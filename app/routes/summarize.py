from flask import Blueprint, request, jsonify
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

summarize_bp = Blueprint('summarize', __name__)

env_path = Path(__file__).resolve().parents[3] / '.env.local'
load_dotenv(dotenv_path=env_path)

GEMINI_KEY = os.environ.get("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

@summarize_bp.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        prompt = f"""
        Create a professional, well-structured summary with the following exact format:

        ## Overview
        [2-3 sentence concise overview in paragraph form]

        ## Key Insights
        • [First key point - full sentence]
        • [Second key point - full sentence]
        • [Third key point - full sentence]

        ## Technical Implementation
        • [First technical detail]
        • [Second technical detail]

        ## Impact & Benefits
        • [First benefit]
        • [Second benefit]

        ## Conclusion
        [Brief concluding paragraph]

        Text to analyze:
        {text}

        Formatting Rules:
        - Use ## for section headers
        - Use • for bullet points (not -)
        - All bullet points must be complete sentences
        - Each bullet point must be on its own line
        - Maintain original technical terminology
        - No introductory/closing phrases
        - Keep professional academic tone
        - Ensure consistent spacing between sections
        """
        
        response = model.generate_content(prompt)
        summary = response.text
        
        # Final cleanup
        summary = summary.replace('•', '•')  # Ensure consistent bullet points
        return jsonify({'summary': summary})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500