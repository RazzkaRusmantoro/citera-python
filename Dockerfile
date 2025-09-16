FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Install spaCy small English model
RUN python -m spacy download en_core_web_sm

# Pre-download Hugging Face model into image to avoid runtime caching issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Set Hugging Face cache to a writable location inside container
ENV HF_HOME=/app/hf_cache

# Expose app port
EXPOSE 5000

# Run the app
CMD ["python", "main.py"]
