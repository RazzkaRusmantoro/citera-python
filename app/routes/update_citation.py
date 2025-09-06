# server/app/routes/updatecitation.py
from flask import Blueprint, request, jsonify
from openai import OpenAI
import os
import traceback

update_citation_bp = Blueprint('update_citation', __name__)

# debug print for env var
print("Initializing OpenAI client...")
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    print("WARNING: OPENAI_KEY not found in environment!")
client = OpenAI(api_key=api_key)


@update_citation_bp.route('/update-citation', methods=['POST'])
def generate_updated_citation():
    print("\n=== Incoming /update-citation request ===")
    try:
        data = request.json
        print("Raw JSON received:", data)
    except Exception as e:
        print("Error parsing JSON:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Invalid JSON"}), 400

    metadata = data.get('metadata', {})
    style = data.get('style')
    url = data.get('url')

    print("Metadata:", metadata)
    print("Style:", style)
    print("URL:", url)

    if not metadata or not style or not url:
        print("Missing field detected!")
        return jsonify({"error": "Missing metadata, style, or url"}), 400

    prompt = f"""
    You are an expert academic citation generator.

    Task:
    - Generate a single, fully formatted citation in strict {style} style using the metadata below.
    - Follow the official {style} guidelines precisely, including punctuation, italics, and order.
    - Gracefully handle missing or unknown data according to {style} rules (e.g., use "n.d." for no date, start with title if author unknown).
    - Use the correct terminology for access/retrieval dates based on {style} (e.g., "Accessed" vs "Retrieved").
    - Output ONLY the citation text. No explanations, comments, or extra formatting.

    Metadata:
    Title: {metadata.get('title', 'Unknown Title')}
    Author: {metadata.get('author', None)}
    Publication Date: {metadata.get('pub_date', None)}
    URL: {url}
    Access Date: {metadata.get('access_date', None)}
    """

    print("Prompt to OpenAI:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate academic citations accurately."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        print("Raw OpenAI response:", response)

        citation = response.choices[0].message.content.strip()
        print("Generated citation:", citation)

        return jsonify({"citation": citation})

    except Exception as e:
        print("Error during OpenAI call:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"OpenAI API error: {e}"}), 500
