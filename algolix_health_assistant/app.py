from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, url_for, redirect

from src.helper import download_hugging_face_embeddings

import os
from pathlib import Path

APP_FILE = Path(__file__).resolve()
MEDICAL_CHATBOT_DIR = APP_FILE.parent.parent
TEMPLATE_DIR = (MEDICAL_CHATBOT_DIR / "algolix_health_assistant" /  "templates").resolve()

print(f"[Flask] Using templates dir: {TEMPLATE_DIR}")

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "algolix-health-assistant"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

system_promt = (
    "You are a helpful AI assistant that helps people find information."
    "Use the following pieces of context to answer the question at the end."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "Answer concisely."
    "\n\n"
    "context: {context}"
)

promt = ChatPromptTemplate.from_messages(
    [
        ("system", system_promt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, promt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def home():
    return render_template('first.html')

@app.route('/signup')
def signup_page():
    
    return render_template('signup.html') 

@app.route('/signup', methods=['POST'])
def signup():
    
    return redirect(url_for('login_page')) 

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    
    email = request.form.get('email')
    password = request.form.get('password')

    
    if email == "test@example.com" and password == "password":
        # If login is successful, return JSON with a success flag and redirect URL
        return jsonify({"success": True, "redirect_url": url_for('ui_page')})
    else:
        
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route('/ui') 
def ui_page():
    return render_template('ui.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(f"User Input: {user_input}")
    
    response = rag_chain.invoke({"input": user_input})
    
    answer = response.get("answer", "No answer found.")
    print(f"AI Response: {answer}")
    
    return str(answer)

if __name__ == '__main__':
    app.run(debug=True)
