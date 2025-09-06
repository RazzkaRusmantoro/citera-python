from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
from keybert import KeyBERT
import spacy
import requests
from openai import OpenAI
import json
import os
from pathlib import Path
from dotenv import load_dotenv

cite_bp = Blueprint('cite', __name__)

# Initialize models once
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT('all-MiniLM-L3-v2')
sbert_model = SentenceTransformer('all-MiniLM-L3-v2')

# Load environment variables
env_path = Path(__file__).resolve().parents[3] / '.env.local'
load_dotenv(dotenv_path=env_path)

OPENAI_KEY = os.environ.get("OPENAI_KEY")
SEMANTIC_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

@cite_bp.route('/cite-openai', methods=['POST'])
def cite():
    try:
        print("API endpoint called")  # Debugging
        data = request.json
        sentences = data.get('sentences', [])
        
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        print(f"Received {len(sentences)} sentences")  # Debugging

        # 1. Extract keywords from all sentences
        combined_text = ' '.join(s['text'] for s in sentences)
        search_terms = [kw[0] for kw in 
            kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=5
            )
        ]
        print(f"🔍 Search Keywords: {search_terms}")

        # 2. Fetch papers from Semantic Scholar
        papers = []
        for term in search_terms:
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": term,
                    "limit": 5,
                    "fields": "title,abstract,authors,year,url,citationCount"
                },
                headers={"x-api-key": SEMANTIC_KEY}
            )
            if response.ok:
                papers.extend(p for p in response.json().get('data', []) if p.get('abstract'))

        # Sort papers by citation count to prioritize more influential papers
        papers = sorted(papers, key=lambda x: x.get('citationCount', 0), reverse=True)
        print(f"📄 Found {len(papers)} papers")  # Debugging

        # 3. Match each sentence to paper abstracts (find best match per sentence)
        matched_pairs = []
        essay_embeddings = sbert_model.encode(
            [s['text'] for s in sentences],
            convert_to_tensor=False
        )
        
        # For each sentence, find the best matching paper
        paper_data = []
        for paper in papers:
            abstract_sents = [sent.text.strip() for sent in nlp(paper["abstract"]).sents]
            if not abstract_sents:
                continue
                
            abstract_embeddings = sbert_model.encode(abstract_sents, convert_to_tensor=False)
            author = paper.get("authors", [{}])[0].get("name", "").split()[-1] if paper.get("authors") else "Author"
            year = paper.get("year", "n.d.")
            
            abstract_embeddings = sbert_model.encode(abstract_sents, convert_to_tensor=False)
            paper_data.append({
                "embeddings": abstract_embeddings,
                "sentences": abstract_sents,
                "title": paper["title"],
                "paper_id": paper.get("paperId", ""),
                "citation": f"({author}, {year})" if author and year else None,
                "citation_count": paper.get("citationCount", 0)
            })

        # Find best match for each sentence
        for sent_idx, sentence in enumerate(sentences):
            best_match = None
            best_score = 0.4
            
            for paper in paper_data:
                similarities = util.cos_sim(
                    essay_embeddings[sent_idx:sent_idx+1], 
                    paper["embeddings"]
                )
                max_score = np.max(similarities)
                
                if max_score > best_score:
                    best_score = max_score
                    best_abstract_sent = paper["sentences"][np.argmax(similarities)]
                    
                    best_match = {
                        "original_sentence": sentence['text'],
                        "paragraph_index": sentence['paragraph_index'],
                        "paper_title": paper["title"],
                        "paper_id": paper["paper_id"],
                        "abstract_sentence": best_abstract_sent,
                        "similarity_score": best_score,
                        "citation": paper["citation"],
                        "citation_count": paper["citation_count"]
                    }
            
            if best_match:
                matched_pairs.append(best_match)

        print(f"🤝 Found {len(matched_pairs)} potential matches")  # Debugging
        with open('matched_pairs.json', 'w') as f:
            json.dump(matched_pairs, f, indent=2)
        print("Saved matched_pairs.json")

        # 4. OpenAI verification with lower confidence threshold
        if matched_pairs:
            combined_prompt = """You are an academic citation assistant with strict content filtering rules.

            STAGE 1: CONTENT VALIDATION - REJECT IMMEDIATELY IF:
            - Sentence is shorter than 8 words
            - Is a section header (e.g., "Introduction", "Conclusion", "Methodology")
            - Contains student info (e.g., "Submitted by", "Student ID")
            - Is metadata (e.g., "Page 3", "Figure 1")
            - Is a bullet point or list item
            - Is a reference/citation placeholder

            STAGE 2: CITATION EVALUATION (only for valid academic content):
            - Be very lenient: accept if abstract somewhat supports the general idea
            - Only reject if there's clearly no connection
            - Accept matches with confidence >= 0.5

            OUTPUT REQUIREMENTS:
            - MUST return valid JSON
            - For each valid sentence, return the original sentence and citation
            - DO NOT rewrite any sentences
            - Only return the citation information

            Return JSON format:
            {
            "analysis": [{
                "original_sentence": str,
                "is_valid_content": bool,
                "needs_citation": bool,
                "citation": "(Author, Year)" or null,
                "confidence": float,
                "paper_id": string,
                "paper_title": string
            }]
            }"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "system", 
                    "content": combined_prompt
                }, {
                    "role": "user",
                    "content": json.dumps({"pairs": matched_pairs})
                }],
                temperature=0,
                max_tokens=10000,
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                print("OpenAI analysis received")
                
                # Process results directly from OpenAI analysis
                final_results = [
                    {           
                        "original_sentence": item["original_sentence"],
                        "cited_sentence": f"{item['original_sentence'].rstrip('.!?;')} {item['citation']}.",
                        "paragraph_index": next(
                            (p["paragraph_index"] for p in matched_pairs 
                            if p["original_sentence"] == item["original_sentence"]), 
                            None
                        ),
                        "citation": item["citation"],
                        "confidence": item.get("confidence", 0),
                        "paper_id": item["paper_id"],
                        "paper_title": item["paper_title"],
                    }
                    for item in analysis.get("analysis", [])
                    if item.get("is_valid_content", False) 
                    and item.get("needs_citation", False)
                    and item.get("confidence", 0) >= 0.5
                    and item.get("citation")
                ]

                print("\nFinal results:", final_results)   

                with open('final_results.json', 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                print("Saved final_results.json")
                
                return jsonify({
                    "status": "success",
                    "results": final_results,
                    "debug": {
                        "matches": matched_pairs,
                        "openai_analysis": analysis
                    }
                }), 200
                
            except json.JSONDecodeError as e:
                print(f"OpenAI JSON parse error: {e}")
                return jsonify({
                    "status": "error",
                    "message": "Failed to parse OpenAI response"
                }), 500
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "trace": str(e.__traceback__)
        }), 500