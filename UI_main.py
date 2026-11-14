import streamlit as st
import tempfile
import os
import hashlib

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


load_dotenv()

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚"
)

st.title("📚 AI Study Assistant")
st.write("Upload your study material and ask questions!")


# Initialize models
@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    return embeddings, llm


embeddings, llm = load_models()


# Prompt
prompt = PromptTemplate.from_template("""
You are a helpful AI study assistant.

Context: {context}

Question: {question}

Instructions:
- Explain in simple, student-friendly language.
- Keep answers clear and relevant.
- Use examples if needed.
- Do not make up information.
- If the answer is not available in the context,
  say you couldn't find it in the uploaded document.

Answer:
""")


# PDF Upload
uploaded_file = st.file_uploader(
    "Upload your PDF notes",
    type=["pdf"]
)


# Process uploaded document
if uploaded_file:

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.get("file_hash") != file_hash:

        with st.spinner("Processing your document..."):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(docs)

                if not chunks:
                    st.error("No readable text found in this PDF.")
                    st.stop()

                # Create vector database
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # Store in session
                st.session_state.vector_store = vector_store
                st.session_state.file_hash = file_hash
                st.session_state.messages = []

                st.success("Document processed successfully!")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.success(f"Document ready: {uploaded_file.name}")


# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []


if "vector_store" in st.session_state:

    st.divider()
    st.subheader("💬 Ask Your Questions")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Student question
    question = st.chat_input("Ask something from your PDF...")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Finding your answer..."):

                # Retrieve relevant chunks
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "lambda_mult": 0.5
                    }
                )

                docs = retriever.invoke(question)

                # Create context
                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                # Generate prompt
                final_prompt = prompt.invoke({
                    "context": context,
                    "question": question
                })

                # LLM response
                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("👆 Please upload a PDF to start asking questions.")