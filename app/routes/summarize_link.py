from flask import Blueprint, request, jsonify
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

summarize_link_bp = Blueprint('summarize_link', __name__)

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

def scrape_website_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        
        # Get text from main content areas
        content = ' '.join([p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        content = ' '.join(content.split())  # Remove extra whitespace
        return content[:15000]  # Limit to 15,000 chars to avoid token limits
    except Exception as e:
        raise Exception(f"Failed to scrape website: {str(e)}")

@summarize_link_bp.route('/summarize-link', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = data['url'].strip()
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL provided'}), 400
    
    try:
        # Step 1: Scrape the website content
        text = scrape_website_content(url)
        if not text:
            return jsonify({'error': 'No readable content found on the page'}), 400
        
        # Step 2: Generate summary using Gemini
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
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500