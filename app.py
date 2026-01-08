from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
GOOGLE_API_KEY = "AIzaSyDCqlCjLbX3AI-DxzeSiFX63_Rh7hdtLG0"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
    timeout=30
)
vectorstore = None
retriever = None
rag_chain = None
pdf_loaded = False
def load_single_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    documents = [doc for doc in documents if doc.page_content.strip()]
    print(f" Loaded {len(documents)} pages from PDF")

    return documents


def initialize_rag_system(pdf_path):
    global vectorstore, retriever, rag_chain, pdf_loaded

    docs = load_single_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """
You are an economics assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't have information about that in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    pdf_loaded = True
    print("✅ RAG system initialized successfully")


def ask_question(query):
    if not pdf_loaded:
        return "No PDF loaded.", []

    try:
        docs = retriever.invoke(query)
        response = rag_chain.invoke(query)

        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return answer, docs

    except Exception as e:
        print(" Gemini Error:", e)
        return "Gemini connection failed.", []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    if not pdf_loaded:
        return jsonify({"error": "PDF not loaded"}), 400

    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer, docs = ask_question(question)
    sources = [doc.metadata.get("page", "N/A") for doc in docs]

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "pdf_loaded": pdf_loaded,
        "ready": pdf_loaded
    })
if __name__ == "__main__":

    BASE_DIR = r"C:\Users\nagam\OneDrive\Desktop\newagent"

    pdf_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(" No PDF found in folder:", BASE_DIR)
    else:
        DEFAULT_PDF = os.path.join(BASE_DIR, pdf_files[0])
        print(" Using PDF:", DEFAULT_PDF)

        try:
            initialize_rag_system(DEFAULT_PDF)
            print("PDF auto-loaded successfully")
        except Exception as e:
            print(" Failed to load PDF:", e)

    app.run(debug=True, host="0.0.0.0", port=5000)

