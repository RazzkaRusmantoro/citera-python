# main.py
from flask import Blueprint, request, jsonify, make_response
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from openai import OpenAI
import os

cite_link_bp = Blueprint('cite_link', __name__)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

# Initialize OpenAI client (requires OPENAI_API_KEY in env)
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def fetch_metadata(url):
    print("Fetching metadata...")
    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Title
        title = soup.title.string.strip() if soup.title else "Unknown Title"
        
        # Author
        author = None
        author_meta_tags = [
            {"name": "author"},
            {"property": "article:author"},
            {"name": "twitter:creator"},
            {"itemprop": "author"},
        ]
        for attrs in author_meta_tags:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                author = tag["content"].strip()
                break
        
        # Publication date
        date = None
        date_meta_tags = [
            {"property": "article:published_time"},
            {"name": "pubdate"},
            {"name": "publication_date"},
            {"name": "date"},
            {"itemprop": "datePublished"},
        ]
        for attrs in date_meta_tags:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                date = tag["content"].strip()
                break
        
        return {
            "title": title,
            "author": author or "Unknown Author",
            "pub_date": date or "Unknown Date",
            "access_date": datetime.now().strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        return {
            "error": f"Error fetching metadata: {e}",
            "title": "Unknown Title",
            "author": "Unknown Author",
            "pub_date": "Unknown Date",
            "access_date": datetime.now().strftime("%Y-%m-%d")
        }

def generate_citation(metadata, style):
    # raise exceptions on errors instead of returning string errors
    prompt = f"""
        You are an expert in academic citation styles.

        Generate a **single, accurate** web citation in **{style}** style using the metadata below:

        Title: {metadata['title']}
        Author: {metadata['author']}
        Publication Date: {metadata['pub_date']}
        URL: {metadata['url']}
        Access Date: {metadata['access_date']}

        Rules:
        1. Follow {style} rules exactly, including punctuation and italics.
        2. If author is unknown, follow that style's convention (e.g., start with title for APA/MLA).
        3. If publication date is unknown, use the proper 'no date' format for that style.
        4. Handle retrieval/access dates correctly based on {style} rules.
        5. Return **only the formatted citation**, no extra commentary.
        """


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate academic citations accurately."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    if not response or not response.choices:
        raise ValueError("OpenAI returned no choices")

    return response.choices[0].message.content.strip()


@cite_link_bp.route('/cite-link', methods=['POST'])
def cite_link():
    data = request.get_json()
    url = data.get('url')
    style = data.get('style')

    if not url or not style:
        return make_response(jsonify({"error": "Missing 'url' or 'style' in parameter"}), 400)

    metadata = fetch_metadata(url)
    metadata['url'] = url

    try:
        citation = generate_citation(metadata, style)
    except Exception as e:
        # log the error somewhere if needed
        return make_response(jsonify({"error": str(e)}), 500)

    return jsonify({
        "citation": citation,
        "metadata": metadata
    })
