from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import spacy
import requests
import numpy as np
from openai import OpenAI
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import re

recite_bp = Blueprint('recite', __name__)

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

def get_author_citation(authors, year):
    """Format author citation in APA style"""
    if not authors:
        return f"(Anonymous, {year})" if year else "(Anonymous, n.d.)"
    
    if len(authors) == 1:
        author_name = authors[0].get("name", "").split()[-1] if authors[0].get("name") else "Anonymous"
        return f"({author_name}, {year})" if year else f"({author_name}, n.d.)"
    elif len(authors) == 2:
        last_names = [author.get("name", "").split()[-1] if author.get("name") else "Anonymous" for author in authors[:2]]
        return f"({last_names[0]} & {last_names[1]}, {year})" if year else f"({last_names[0]} & {last_names[1]}, n.d.)"
    else:
        first_author = authors[0].get("name", "").split()[-1] if authors[0].get("name") else "Anonymous"
        return f"({first_author} et al., {year})" if year else f"({first_author} et al., n.d.)"

@recite_bp.route('/recite', methods=['POST'])
def cite():
    try:
        data = request.json
        cited_sentence = data.get('citedSentence', '')
        
        # Extract the current citation from the sentence
        citation_match = re.search(r'(\([^)]+\))\.?$', cited_sentence)
        current_citation = citation_match.group(1) if citation_match else ""
        
        # Extract the actual sentence text without the citation
        base_sentence = re.sub(r'\s*\([^)]*\)\.?$', '', cited_sentence).strip()
        
        # 1. Extract keywords from the sentence with multiple strategies
        keywords = []
        
        # Strategy 1: KeyBERT with different n-grams
        keywords.extend([kw[0] for kw in 
            kw_model.extract_keywords(
                base_sentence,
                keyphrase_ngram_range=(1, 1),
                stop_words='english',
                top_n=2
            )
        ])
        
        keywords.extend([kw[0] for kw in 
            kw_model.extract_keywords(
                base_sentence,
                keyphrase_ngram_range=(2, 2),
                stop_words='english',
                top_n=2
            )
        ])
        
        # Strategy 2: Use spaCy for noun phrases
        doc = nlp(base_sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        keywords.extend(noun_phrases[:3])
        
        # Remove duplicates and empty keywords
        keywords = list(set([kw.strip() for kw in keywords if kw.strip()]))
        
        print("All keywords:", keywords)
        
        # 2. Find alternative papers with multiple search strategies
        all_papers = []
        
        # Search with each keyword
        for term in keywords:
            try:
                response = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": term,
                        "limit": 15,  # Get more papers per keyword
                        "fields": "title,abstract,authors,year,url,paperId,citationCount,publicationVenue"
                    },
                    headers={"x-api-key": SEMANTIC_KEY},
                    timeout=10
                )
                
                if response.ok:
                    papers = response.json().get('data', [])
                    # Filter papers with abstracts and good metadata
                    filtered_papers = [
                        p for p in papers 
                        if (p.get('abstract') and len(p.get('abstract', '')) > 50 and 
                            p.get('paperId') and p.get('title') and
                            p.get('authors') and len(p.get('authors', [])) > 0)
                    ]
                    all_papers.extend(filtered_papers)
                    
            except requests.RequestException as e:
                print(f"Error searching for term '{term}': {e}")
                continue
        
        # If no papers found with keywords, try searching with the whole sentence
        if not all_papers:
            try:
                response = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": base_sentence[:100],  # First 100 chars
                        "limit": 20,
                        "fields": "title,abstract,authors,year,url,paperId,citationCount,publicationVenue"
                    },
                    headers={"x-api-key": SEMANTIC_KEY},
                    timeout=10
                )
                
                if response.ok:
                    papers = response.json().get('data', [])
                    filtered_papers = [
                        p for p in papers 
                        if (p.get('abstract') and len(p.get('abstract', '')) > 50 and 
                            p.get('paperId') and p.get('title') and
                            p.get('authors') and len(p.get('authors', [])) > 0)
                    ]
                    all_papers.extend(filtered_papers)
                    
            except requests.RequestException as e:
                print(f"Error searching with full sentence: {e}")
        
        # Remove duplicates and sort by citation count + recent year
        seen_paper_ids = set()
        unique_papers = []
        
        for paper in all_papers:
            if paper['paperId'] not in seen_paper_ids:
                seen_paper_ids.add(paper['paperId'])
                # Calculate a combined score (citation count + recent year bonus)
                citation_count = paper.get('citationCount', 0)
                year = paper.get('year', 0) or 0
                recent_bonus = max(0, (year - 2010) * 10) if year > 2010 else 0  # Bonus for recent papers
                paper['combined_score'] = citation_count + recent_bonus
                unique_papers.append(paper)
        
        # Sort by combined score
        unique_papers = sorted(unique_papers, key=lambda x: x.get('combined_score', 0), reverse=False)
        
        print(f"Found {len(unique_papers)} unique papers")
        
        # 3. Always return the best match, even if similarity is low
        if unique_papers:
            # Encode the sentence
            sentence_embedding = sbert_model.encode([base_sentence], convert_to_tensor=False)
            
            # Encode paper abstracts (top 10 by combined score)
            paper_abstracts = [paper['abstract'] for paper in unique_papers[:10]]
            paper_embeddings = sbert_model.encode(paper_abstracts, convert_to_tensor=False)
            
            # Calculate similarities
            similarities = util.cos_sim(sentence_embedding, paper_embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_paper = unique_papers[best_match_idx]
            similarity_score = similarities[best_match_idx].item()
            
            # Format the citation
            authors = best_paper.get("authors", [])
            year = best_paper.get("year", "")
            new_citation = get_author_citation(authors, year)
            
            print(f"Best match - Similarity: {similarity_score:.3f}, Citation: {new_citation}")
            
            return jsonify({
                "new_citation": new_citation,
                "new_paper_id": best_paper.get("paperId"),
                "new_paper_title": best_paper.get("title"),
                "similarity_score": similarity_score,
                "confidence": "high" if similarity_score > 0.4 else "medium" if similarity_score > 0.2 else "low"
            })
        
        # If absolutely no papers found (very rare), return the original citation
        return jsonify({
            "new_citation": current_citation,
            "new_paper_id": "",
            "new_paper_title": "",
            "similarity_score": 0,
            "confidence": "none"
        })
        
    except Exception as e:
        print(f"Error in recite: {str(e)}")
        return jsonify({
            "error": str(e),
            "new_citation": current_citation if 'current_citation' in locals() else "",
            "new_paper_id": "",
            "new_paper_title": "",
            "similarity_score": 0,
            "confidence": "error"
        }), 500