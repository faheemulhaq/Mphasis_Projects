import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🤖 Meet Dexter</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Chat with your PDFs using our interactive chatbot, Dexter. Upload your documents, ask questions, and get instant answers!</p>", unsafe_allow_html=True)

    st.markdown("""
    <button onclick="startRecognition()">Activate Voice Commands</button>
    <p id="status"></p>
    <script>
        var recognition;
        function startRecognition() {
            if (!('webkitSpeechRecognition' in window)) {
                document.getElementById('status').innerHTML = 'Your browser does not support voice recognition.';
                return;
            }

            recognition = new webkitSpeechRecognition();
            recognition.lang = 'en-US';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = function() {
                document.getElementById('status').innerHTML = 'Voice recognition started. Try speaking into the microphone.';
            };

            recognition.onerror = function(event) {
                document.getElementById('status').innerHTML = 'Error occurred in recognition: ' + event.error;
            };

            recognition.onend = function() {
                document.getElementById('status').innerHTML = 'Voice recognition ended.';
            };

            recognition.onresult = function(event) {
                var userQuestion = event.results[0][0].transcript;
                document.getElementById('status').innerHTML = 'You said: ' + userQuestion;
                document.getElementById('user-question').value = userQuestion;
                document.getElementById('ask-button').click();
            };

            recognition.start();
        }
    </script>
    """, unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files", placeholder="Type your question here...", key="user-question")

    if st.button("Ask Dexter", key="ask-button"):
        with st.spinner("Dexter is thinking..."):
            response = user_input(user_question)
            st.markdown(f"<div class='dexter-bubble'><p><strong>Dexter:</strong> {response}</p></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu")
        st.write("Upload and process your PDF files here.")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            with st.spinner("Dexter is processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete. You can now ask questions!")

# Run the main function
if __name__ == "__main__":
    main()
