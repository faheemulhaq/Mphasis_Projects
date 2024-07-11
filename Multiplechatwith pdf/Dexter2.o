import streamlit as st
from PyPDF2 import PdfReader
import docx
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
import openpyxl

load_dotenv()

# Function to extract text from PDF files
@st.cache_data(show_spinner=False)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Word files
@st.cache_data(show_spinner=False)
def get_docx_text(docx_files):
    text = ""
    for doc in docx_files:
        doc_reader = docx.Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text + "\n"
    return text

# Function to extract text from Excel files
@st.cache_data(show_spinner=False)
def get_excel_text(excel_files):
    text = ""
    for excel in excel_files:
        df = pd.read_excel(excel)
        text += df.to_string(index=False)
    return text

# Function to split text into chunks
@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store using FAISS
@st.cache_data(show_spinner=False)
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input question and retrieve answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to convert text to speech using pyttsx3
def text_to_speech(text, mute=False):
    if not mute:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        engine.setProperty('volume', 1.0)  # Volume 0-1
        engine.say(text)
        engine.runAndWait()

# Function to handle voice commands
def get_voice_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        st.info("Listening for your question...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        st.success(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        st.error("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        st.error("Could not request results; check your network connection.")
        return ""

# Main Streamlit application
def main():
    st.set_page_config(page_title="Document Chatbot", layout="wide")
    st.title("Document Chatbot")

    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
            padding: 2rem;
            font-family: Arial, sans-serif;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2E7D32,#1B5E20);
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #2E7D32;
            border: None;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #1B5E56;
            border: None;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 3rem;
            color: #2E7D32;
        }
        .header p {
            font-size: 1.2rem;
            color: #666;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .question-input {
            width: 80%;
            margin-bottom: 1rem;
        }
        .response {
            width: 80%;
            padding: 1rem;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        .response h3 {
            color: #2E7D32;
        }
        .speak-btn {
            background-color: #2E7D32; /* Green color */
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .speak-btn:hover {
            background-color: #1B5E56; /* Darker green on hover */
        }
        .mute-btn {
            background-color: #FF5722; /* Orange color */
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .mute-btn:hover {
            background-color: #E64A19; /* Darker orange on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="header">
        <h1> Chat with Dexter 🤖</h1>
        <p>Upload your documents, process them, and ask questions directly from the content.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for various document types
    doc_files = st.file_uploader("Upload your Documents", accept_multiple_files=True, type=["pdf", "docx", "xlsx"])

    if doc_files:
        # Process documents and store vector store
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = ""
                    pdf_docs = [doc for doc in doc_files if doc.type == "application/pdf"]
                    docx_files = [doc for doc in doc_files if doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
                    excel_files = [doc for doc in doc_files if doc.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]

                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if docx_files:
                        raw_text += get_docx_text(docx_files)
                    if excel_files:
                        raw_text += get_excel_text(excel_files)

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. You can now ask questions!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # State management for mute functionality
        if "mute" not in st.session_state:
            st.session_state.mute = False

        def toggle_mute():
            st.session_state.mute = not st.session_state.mute

        # Function to handle user question input
        def handle_user_question(user_question):
            if user_question:
                with st.spinner("Fetching answer..."):
                    try:
                        response = user_input(user_question)
                        st.markdown(f"""
                        <div class="response">
                            <h3>Reply:</h3>
                            <p>{response}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        text_to_speech(response, mute=st.session_state.mute)  # Read out the response if not muted
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        # Input for user question (text)
        user_question = st.text_input("Enter your question:", key="question")

        # Toggle button to mute/unmute
        if st.button("Mute" if st.session_state.mute else "Unmute", key="mute-btn", on_click=toggle_mute):
            st.session_state.mute = not st.session_state.mute

        if st.button("Ask"):
            handle_user_question(user_question)

        # Input for user question (voice)
        if st.button("Ask using Voice", key="voice-btn"):
            voice_command = get_voice_command()
            if voice_command:
                handle_user_question(voice_command)

if __name__ == "__main__":
    main()
