🌍 AURA – Economy AI Assistant (RAG-Based)

A full-stack Retrieval-Augmented Generation (RAG) AI assistant that provides economics and loan-related insights strictly from PDF documents.
It combines a Gemini-powered chatbot with an interactive 3D global macroeconomic dashboard built using Three.js.

✨ Features
🤖 AI-powered economy & loan advisory chatbot
📄 PDF-based knowledge grounding (no hallucinations)
🔍 FAISS vector search for semantic retrieval
🧠 Gemini LLM integration
🌍 Interactive 3D globe with country analytics
📊 GDP, growth, rank & export visualization
⚙️ Flask backend API
🎨 Modern glassmorphism frontend UI

🧱 Tech Stack
Backend
Python
Flask
Flask-CORS
LangChain
FAISS
Sentence Transformers
Gemini API
Frontend
HTML
Tailwind CSS
JavaScript
Three.js

🧠 How It Works (RAG Flow)
PDF document is automatically loaded on server start
Text is split into chunks and embedded
FAISS stores embeddings for similarity search
User asks a question via chat UI
Relevant document chunks are retrieved
Context + question sent to Gemini
Gemini answers only using document context
Response is displayed in the AURA chat panel
▶️ How to Run:
1️⃣ Clone the Repository
2️⃣ Create & Activate Virtual Environment
3️⃣ Install Required Dependencies:
pip install flask flask-cors \
langchain langchain-community langchain-core langchain-text-splitters \
langchain-google-genai \
faiss-cpu \
sentence-transformers huggingface-hub \
pypdf \
python-dotenv
4️⃣ Add Environment Variables
5️⃣ Add Your PDF
6️⃣ Run the Application
