AI-Powered Text Summarizer & Sentiment Analyzer
✨ Project Overview

The AI-Powered Text Summarizer & Sentiment Analyzer is a robust, full-stack web application designed to process lengthy articles and documents, providing concise summaries and insightful sentiment analysis. Leveraging advanced Natural Language Processing (NLP) models from Hugging Face's transformers library, this tool offers both abstractive summarization and a basic extractive fallback, along with sentiment detection for both original and summarized texts.

Developed as a Flask backend with a modern HTML/CSS/JavaScript frontend, this project serves as a practical demonstration of integrating sophisticated AI capabilities into a user-friendly web interface.

🎯 Task Objective

The primary objective was to build a comprehensive text summarization solution, enhance it with sentiment analysis, and ensure it can handle various text lengths, including very long documents, through intelligent chunking and fallback mechanisms. The project also focuses on creating an intuitive and visually appealing user experience.

🔍 Key Features

✅ Core AI Functionality:

Abstractive Summarization: Utilizes the facebook/bart-large-cnn model to generate coherent and concise summaries that capture the essence of the input text, even rephrasing content.

Long Text Handling (Chunking): Implements a "map-reduce" style chunking strategy for texts exceeding the model's maximum input length. It splits text into manageable segments, summarizes each, and then concatenates the mini-summaries, with an optional second-pass summarization for extremely long combined summaries.

Extractive Fallback: Provides a basic extractive summarization using NLTK (based on sentence scoring) as a fallback mechanism if the abstractive model fails to load or generate a summary.

Sentiment Analysis: Employs the cardiffnlp/twitter-roberta-base-sentiment-latest model to determine the emotional tone (Positive, Negative, Neutral) of both the original input text and the generated summary, complete with confidence scores.

✅ Data & Performance Management:

Model Caching: Efficiently loads and caches large language models (BART, RoBERTa) in memory using cachetools.TTLCache to significantly reduce latency on subsequent requests with the same input.

Asynchronous Processing (Implicit): Leverages Flask's threaded=True and the ThreadPoolExecutor for background processing of summarization and sentiment analysis tasks, ensuring the web UI remains responsive during model inference.

Robust Error Handling: Comprehensive try-except blocks and detailed logging ensure the application handles model loading failures, processing errors, and invalid inputs gracefully.

Rate Limiting: Basic in-memory rate limiting is implemented to prevent abuse and manage server load.

Logging: Detailed logging to both console and api.log file for monitoring and debugging.

⚙️ Tech Stack

Languages: Python, HTML5, CSS3, JavaScript

Web Framework: Flask

Text Summarization: transformers (Hugging Face - facebook/bart-large-cnn)

Sentiment Analysis: transformers (Hugging Face - cardiffnlp/twitter-roberta-base-sentiment-latest)

NLP & Preprocessing: nltk (for sentence tokenization, stopwords)

Performance & Utilities: torch, cachetools, concurrent.futures, typing, logging, time

UI/Styling: Tailwind CSS

Development Environment: Local machine (with option for GPU inference if CUDA is available)

🎯 Learning Outcomes

Through this project, I gained a comprehensive understanding and practical experience in:

Full-Stack AI Development: Building and integrating a machine learning backend with a responsive web frontend.

Advanced NLP Techniques: Implementing abstractive summarization, handling long texts with chunking strategies, and performing sentiment analysis.

Model Management & Optimization: Efficiently loading, caching, and utilizing large transformer models, including device management (CPU/GPU).

Robust API Design: Implementing features like rate limiting, comprehensive error handling, and structured JSON responses for a production-ready API.

Frontend Development with Modern CSS: Crafting an intuitive and visually appealing user interface using custom CSS and Tailwind CSS for responsive design and "glassmorphism" effects.

Inter-Process Communication: Understanding and implementing client-server communication via AJAX (Fetch API) between the frontend and Flask API.

🚀 Setup & Installation

Follow these steps to get the AI-Powered Text Summarizer & Sentiment Analyzer up and running on your local machine.

Prerequisites

Python 3.8+ installed (Ensure python and pip commands work in your terminal).

Git (for cloning the repository).

Clone the Repository

git clone https://github.com/mishal4583/Generative-AI-Text-Summarizer.git
cd Generative-AI-Text-Summarizer

Project Structure

Ensure your project directory looks like this:

Generative-AI-Text-Summarizer/
├── images/
│   └── bg1.png          # Background image for the UI
├── api.log              # Log file (will be created on first run)
├── index.html           # Frontend HTML, CSS (Tailwind), and JavaScript
└── text_summarizer_app.py # Python Flask Backend with AI logic

Make sure bg1.png is placed inside the images folder.

Install Python Dependencies

It's recommended to use a virtual environment.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install Flask transformers flask-cors cachetools nltk torch

Download NLTK Data

The nltk library requires specific data for sentence tokenization. Run these commands once:

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

You will see download progress. Once completed, you can proceed.

Run the Flask Backend

Open your terminal in the Generative-AI-Text-Summarizer directory and run:

python text_summarizer_app.py

The first time you run this, it will download the large AI models (bart-large-cnn and roberta-base), which can take several minutes depending on your internet speed.

Keep this terminal window open; the Flask server needs to be running for the frontend to work.

You should see output like * Running on http://127.0.0.1:5000

Open the Frontend

With the Flask backend running, open the index.html file in your web browser.

You can usually just double-click index.html in your file explorer.

Alternatively, you can use a simple HTTP server (e.g., Python's built-in one) from your project directory:

# In a NEW terminal window, navigate to the project directory
python -m http.server 8000
# Then open your browser to http://localhost:8000/index.html

Now you can paste text into the input field, click "SUMMARIZE & ANALYZE," and see the AI-powered results!
