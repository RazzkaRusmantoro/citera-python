from flask import Blueprint, request, jsonify
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
import json

source_checker_bp = Blueprint('source_checker', __name__)

env_path = Path(__file__).resolve().parents[3] / '.env.local'
load_dotenv(dotenv_path=env_path)

GEMINI_KEY = os.environ.get("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_article_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        
        # Get text from main content areas
        article_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'article'])])
        
        # Get metadata
        title = soup.title.string if soup.title else ''
        description = soup.find('meta', attrs={'name': 'description'})
        description = description['content'] if description else ''
        
        return {
            'title': title,
            'description': description,
            'text': article_text.strip(),
            'domain': urlparse(url).netloc
        }
    except Exception as e:
        raise Exception(f"Error extracting content: {str(e)}")

@source_checker_bp.route('/source_checker', methods=['POST'])
def check():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url or not is_valid_url(url):
            return jsonify({
                'url': url,
                'domain': urlparse(url).netloc if url else '',
                'analysis': {'error': 'Invalid or missing URL'}
            }), 400
        
        try:
            extracted_content = extract_article_text(url)
        except Exception as e:
            return jsonify({
                'url': url,
                'domain': urlparse(url).netloc,
                'analysis': {'error': f'Failed to extract content: {str(e)}'}
            }), 400

        system_prompt = """You are an academic-grade source credibility evaluator..."""  # Keep your existing prompt
        
        prompt_template = f"""Analyze the credibility of the following article content. Provide STRICTLY ONLY a JSON output with:
        {{
            "overall_score": 0-100,
            "summary": "string",
            "breakdown": {{
                "factual_accuracy": 0-100,
                "source_reputation": 0-100,
                "author_expertise": 0-100,
                "content_bias": 0-100,
                "transparency": 0-100
            }},
            "warnings": ["string"],
            "recommendations": ["string"]
        }}

        Article Metadata:
        - Title: {extracted_content['title']}
        - Domain: {extracted_content['domain']}
        - Description: {extracted_content['description']}

        Content:
        {extracted_content['text'][:20000]}  # Increased limit
        """
        
        try:
            response = model.generate_content([system_prompt, prompt_template])
            response_text = response.text
            
            # Clean the response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            analysis = json.loads(response_text)
            
            # Validate the response structure
            required_fields = ['overall_score', 'summary', 'breakdown', 'warnings', 'recommendations']
            if not all(field in analysis for field in required_fields):
                raise ValueError("Missing required fields in analysis")
                
            required_breakdown = ['factual_accuracy', 'source_reputation', 'author_expertise', 'content_bias', 'transparency']
            if not all(field in analysis['breakdown'] for field in required_breakdown):
                raise ValueError("Missing required breakdown fields")
                
        except Exception as e:
            analysis = {
                'error': 'Failed to analyze content',
                'details': str(e)
            }
        
        response_data = {
            'url': url,
            'domain': extracted_content['domain'],
            'analysis': analysis
        }
        
        print("\n=== API Response ===")
        print(json.dumps(response_data, indent=2))
        print("===================\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_response = {
            'url': url if 'url' in locals() else '',
            'domain': urlparse(url).netloc if 'url' in locals() else '',
            'analysis': {'error': f'An error occurred: {str(e)}'}
        }
        print("\n=== Error Response ===")
        print(json.dumps(error_response, indent=2))
        print("=====================\n")
        return jsonify(error_response), 500