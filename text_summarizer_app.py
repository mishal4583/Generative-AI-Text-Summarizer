# text_summarizer_app.py

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
from cachetools import TTLCache
import time
from typing import Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.warning("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')

# Configuration
CONFIG = {
    "MAX_INPUT_LENGTH": 1024, # Max tokens the model can handle
    "SUMMARY_MIN_LENGTH": 30,
    "SUMMARY_MAX_LENGTH": 150,
    "CHUNK_OVERLAP": 100, # Number of tokens to overlap between chunks
    "CACHE_TTL": 300,    # 5 minutes
    "CACHE_SIZE": 1000,
    "MODELS": {
        "summarization": "facebook/bart-large-cnn",
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        # Extractive model is not directly loaded as a pipeline for fallback; NLTK handles it.
        # "extractive": "distilbert-base-uncased"
    }
}

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=4)
cache = TTLCache(maxsize=CONFIG['CACHE_SIZE'], ttl=CONFIG['CACHE_TTL'])

class ModelManager:
    """Advanced model management with fallback strategies"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.models = {
            "summarization": self._load_summarization_model(),
            "sentiment": self._load_sentiment_model(),
        }
        
        # Tokenizers specifically for direct model calls (like generate_summary's internal use)
        self.tokenizers = {
            "summarization": AutoTokenizer.from_pretrained(CONFIG['MODELS']['summarization']),
            "sentiment": AutoTokenizer.from_pretrained(CONFIG['MODELS']['sentiment'])
        }
    
    def _load_summarization_model(self):
        try:
            model = pipeline(
                "summarization",
                model=CONFIG['MODELS']['summarization'],
                device=self.device
            )
            logger.info("Loaded abstractive summarization model")
            return model
        except Exception as e:
            logger.error(f"Failed to load abstractive model: {e}", exc_info=True)
            return None
    
    def _load_sentiment_model(self):
        try:
            model = pipeline(
                "sentiment-analysis",
                model=CONFIG['MODELS']['sentiment'],
                device=self.device
            )
            logger.info("Loaded sentiment analysis model")
            return model
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}", exc_info=True)
            return None
    
    # Removed _load_extractive_model as NLTK is used for simple extractive fallback

model_manager = ModelManager()

def rate_limit_exceeded(client_ip: str) -> bool:
    """Basic rate limiting implementation"""
    current_time = time.time()
    if client_ip not in cache:
        cache[client_ip] = {'count': 1, 'timestamp': current_time}
        return False
    
    time_diff = current_time - cache[client_ip]['timestamp']
    if time_diff > 60:  # Reset counter after 1 minute
        cache[client_ip] = {'count': 1, 'timestamp': current_time}
        return False
    
    cache[client_ip]['count'] += 1
    return cache[client_ip]['count'] > 30  # Limit to 30 requests per minute

def generate_summary(text: str, model, tokenizer) -> str:
    """Generate summary with fallback strategies"""
    if model is None:
        logger.error("Summarization model is not loaded. Falling back to extractive summary.")
        return fallback_extractive_summary(text)

    # Tokenize the entire text to check its length in tokens
    # Using the summarization tokenizer as its MAX_INPUT_LENGTH is relevant here
    tokenized_text = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = tokenized_text.input_ids[0]
    num_tokens = len(input_ids)

    # Check if text needs chunking based on model's max input length
    # It's better to use tokens than words for precise model limits
    if num_tokens > CONFIG['MAX_INPUT_LENGTH']:
        logger.info(f"Text too long ({num_tokens} tokens), chunking for summarization.")
        return chunk_and_summarize(text, model, tokenizer)
    
    try:
        summary_ids = model.model.generate(
            input_ids.unsqueeze(0).to(model.device), # Add batch dimension and move to device
            max_length=CONFIG['SUMMARY_MAX_LENGTH'],
            min_length=CONFIG['SUMMARY_MIN_LENGTH'],
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Abstractive summarization failed: {e}", exc_info=True)
        return fallback_extractive_summary(text)

def chunk_and_summarize(text: str, model, tokenizer) -> str:
    """
    Handles long documents by splitting them into chunks, summarizing each,
    and then concatenating the summaries.
    For extremely long texts, a recursive summarization of the summaries
    would be needed, but this version focuses on processing large inputs.
    """
    if model is None:
        logger.error("Summarization model not loaded for chunking. Cannot perform chunked summarization.")
        return "Chunking summarization failed due to missing model."

    # Split text into sentences
    sentences = sent_tokenize(text)
    
    current_chunk = []
    chunk_summaries = []
    
    for sentence in sentences:
        # Check if adding the next sentence exceeds max_length
        # Add a small buffer for safety and overlap
        temp_chunk_str = tokenizer.convert_tokens_to_string(current_chunk + tokenizer.tokenize(sentence))
        if len(tokenizer(temp_chunk_str).input_ids) > CONFIG['MAX_INPUT_LENGTH'] - 50: # -50 for buffer
            if current_chunk: # Summarize if chunk is not empty
                chunk_text = tokenizer.decode(tokenizer.encode(tokenizer.convert_tokens_to_string(current_chunk), add_special_tokens=False))
                try:
                    summary_result = model(chunk_text, max_length=CONFIG['SUMMARY_MAX_LENGTH'], min_length=CONFIG['SUMMARY_MIN_LENGTH'], do_sample=False)
                    chunk_summaries.append(summary_result[0]['summary_text'])
                except Exception as e:
                    logger.error(f"Error summarizing chunk: {e}", exc_info=True)
                    chunk_summaries.append(f"Error summarizing chunk: {str(e)}")
            current_chunk = tokenizer.tokenize(sentence) # Start new chunk with current sentence
        else:
            current_chunk.extend(tokenizer.tokenize(sentence)) # Add sentence to current chunk
    
    # Summarize the last chunk
    if current_chunk:
        chunk_text = tokenizer.decode(tokenizer.encode(tokenizer.convert_tokens_to_string(current_chunk), add_special_tokens=False))
        try:
            summary_result = model(chunk_text, max_length=CONFIG['SUMMARY_MAX_LENGTH'], min_length=CONFIG['SUMMARY_MIN_LENGTH'], do_sample=False)
            chunk_summaries.append(summary_result[0]['summary_text'])
        except Exception as e:
            logger.error(f"Error summarizing final chunk: {e}", exc_info=True)
            chunk_summaries.append(f"Error summarizing final chunk: {str(e)}")
            
    final_summary = " ".join(chunk_summaries).strip()

    # If the combined summary is still very long, re-summarize it (recursive summarization)
    # This is a common pattern for "summarizing a summary"
    if tokenizer(final_summary).input_ids and len(tokenizer(final_summary).input_ids) > CONFIG['MAX_INPUT_LENGTH']:
        logger.info(f"Combined summary is still long ({len(tokenizer(final_summary).input_ids)} tokens), performing second-pass summarization.")
        try:
            second_pass_summary_result = model(final_summary, max_length=CONFIG['SUMMARY_MAX_LENGTH'], min_length=CONFIG['SUMMARY_MIN_LENGTH'], do_sample=False)
            return second_pass_summary_result[0]['summary_text']
        except Exception as e:
            logger.error(f"Error during second-pass summarization: {e}", exc_info=True)
            return "Second-pass summarization failed. " + final_summary
    
    return final_summary if final_summary else "No summary could be generated for the long text."


def fallback_extractive_summary(text: str) -> str:
    """
    Fallback to a simple extractive summarization using NLTK (sentence tokenization)
    and selecting the first few sentences.
    This is a basic fallback and can be enhanced with more sophisticated
    extractive methods if needed (e.g., TextRank).
    """
    logger.info("Falling back to extractive summarization.")
    sentences = sent_tokenize(text)
    
    # Simple strategy: take the first few sentences that fit within a reasonable length.
    # Adjust number of sentences based on desired length for extractive.
    # Aim for roughly 3-5 sentences for a short extractive summary.
    extractive_summary_sentences = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence.split()) <= 150: # max 150 words
            extractive_summary_sentences.append(sentence)
            current_length += len(sentence.split())
        else:
            break
            
    if not extractive_summary_sentences:
        # If no sentences fit the length, just take the first sentence as a last resort
        return sentences[0] if sentences else "No summary available (extractive fallback)."
        
    return " ".join(extractive_summary_sentences)

def analyze_sentiment(text: str, model, tokenizer=None) -> Dict[str, Any]:
    """Advanced sentiment analysis with confidence scores"""
    try:
        if model is None:
            logger.error("Sentiment analysis model is not loaded.")
            return {"label": "ERROR", "score": 0.0}

        if not text.strip():
            return {"label": "N/A", "score": 0.0}

        # Call the pipeline directly with the text string.
        # The pipeline handles its own tokenization and truncation.
        # Limit text size explicitly for the sentiment model input
        sentiment_results = model(text[:512]) 

        if not sentiment_results:
            return {"label": "N/A", "score": 0.0}

        first_result = sentiment_results[0]
        raw_label = first_result['label']
        score = first_result['score']
        
        # Map raw labels to POSITIVE, NEGATIVE, NEUTRAL for frontend consistency
        if "positive" in raw_label.lower() or raw_label == 'LABEL_2':
            final_label = "POSITIVE"
        elif "negative" in raw_label.lower() or raw_label == 'LABEL_0':
            final_label = "NEGATIVE"
        else: # Catches 'neutral' or 'LABEL_1'
            final_label = "NEUTRAL"

        return {
            "label": final_label,
            "score": score
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
        return {"label": "ERROR", "score": 0.0}

@app.before_request
def before_request():
    """Pre-request processing"""
    client_ip = request.remote_addr
    if rate_limit_exceeded(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "/summarize": "POST text for summarization and sentiment analysis",
            "/health": "Check service health"
        },
        "models_loaded": {
            "summarization": model_manager.models['summarization'] is not None,
            "sentiment": model_manager.models['sentiment'] is not None
        }
    })

@app.route('/health')
def health_check():
    """Advanced health check endpoint"""
    # Note: _load_extractive_model is not used, so it's not checked here
    models_ok = (model_manager.models['summarization'] is not None) and \
                (model_manager.models['sentiment'] is not None)
    
    return jsonify({
        "status": "healthy" if models_ok else "degraded",
        "models": {
            "summarization": "loaded" if model_manager.models['summarization'] else "failed",
            "sentiment": "loaded" if model_manager.models['sentiment'] else "failed",
            "extractive_fallback_available": True # NLTK is always available if installed
        },
        "device": model_manager.device,
        "cache_size": len(cache)
    }), 200 if models_ok else 503

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Advanced text processing endpoint"""
    start_time = time.time()
    
    # Input validation
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({
            "original_text": "",
            "summary": "",
            "original_sentiment": {"label": "N/A", "score": 0.0},
            "summary_sentiment": {"label": "N/A", "score": 0.0}
        }), 200
    
    # Check cache
    cache_key = hash(text)
    if cache_key in cache:
        logger.info("Returning cached result")
        return jsonify(cache[cache_key])
    
    try:
        # Generate summary
        summary = generate_summary(
            text,
            model_manager.models['summarization'],
            model_manager.tokenizers['summarization']
        )
        
        # Analyze sentiment for both original text and summary
        original_sentiment = analyze_sentiment(
            text,
            model_manager.models['sentiment']
        )
        
        # Ensure summary is not empty before analyzing sentiment
        summary_to_analyze = summary if summary.strip() else ""
        summary_sentiment = analyze_sentiment(
            summary_to_analyze,
            model_manager.models['sentiment']
        )

        result = {
            "original_text": text,
            "summary": summary,
            "original_sentiment": original_sentiment,
            "summary_sentiment": summary_sentiment,
            "metadata": {
                "model": CONFIG['MODELS']['summarization'] if "chunk summarization" not in summary else "Chunked Summarization",
                "processing_time": time.time() - start_time,
                "chars_processed": len(text)
            }
        }
        
        # Cache result
        cache[cache_key] = result
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
