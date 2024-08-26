#conda activate "C:\Users\HP\Desktop\Chatbot with Flask\Env1"

from flask import Flask, render_template, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Retrieve API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """If the question is general, such as:
    - "Hi"
    - "Hello"
    - "How are you?"
    - "What’s up?"
    - "Can you help me?"
    - "What’s your name?"
    - "Tell me a joke."
    - "What time is it?"
    - "Where are you located?"
    - "How do I contact support?"
    - "What is your purpose?"
    - "Can you tell me about yourself?"
    - "What can you do?"
    - "How can I get started?"  
    - "What are your hours of operation?"
    - "What is your favorite color?"
    - "Do you have any recommendations?"
    - "Can you provide some information?"
    - "How do I use this service?"
    - "What are the latest updates?"
    - "Can you explain something to me?"
    - "What’s new?"
    - "Do you have any news?"
    - "What’s the best way to reach you?"

    Provide a friendly and relevant response for these questions.

    If the question is specific to the provided context, answer it based on the context only. Please provide the most accurate response based on the question. If an answer cannot be found in the context, provide the closest related answer.

    <context>
    {context}
    <context>
    Questions: {input}"""
)

# Function to initialize vector embeddings
def vector_embedding():
    if "vectors" not in app.config:
        app.config["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        app.config["loader"] = PyPDFDirectoryLoader("./data")
        app.config["docs"] = app.config["loader"].load()
        app.config["text_splitter"] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        app.config["final_documents"] = app.config["text_splitter"].split_documents(app.config["docs"][:20])
        app.config["vectors"] = FAISS.from_documents(app.config["final_documents"], app.config["embeddings"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    prompt1 = request.json.get('message')

    vector_embedding()  # Ensure vectors are initialized
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = app.config["vectors"].as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()  # Start timing the response generation
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start 

    return jsonify({'answer': response['answer']})

if __name__ == '__main__':
    app.run(debug=True)
