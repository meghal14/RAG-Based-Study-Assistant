# RAG-Based Study Assistant

## About the Project

RAG-Based Study Assistant is an AI-powered application that helps students understand their study materials by asking questions directly from uploaded documents.

Users can upload documents such as DBMS notes, programming notes, or other study materials and ask questions. The system retrieves relevant information from the uploaded documents and uses an LLM to generate clear, context-based answers.

The core RAG pipeline is developed by me using LangChain. GPT assistance was used to create a simple Streamlit user interface.

## Features

- Upload study materials and documents.
- Extract and process document content.
- Generate embeddings using Hugging Face.
- Store document embeddings in ChromaDB.
- Retrieve relevant information using MMR (Maximum Marginal Relevance).
- Generate answers using Groq LLM.
- Provide context-based answers to student questions.
- Simple Streamlit interface.

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings (`intfloat/e5-large-v2`)
- ChromaDB (Vector Database)
- Groq API (`openai/gpt-oss-20b`)
- Streamlit
- Python-dotenv

## How It Works

1. User uploads a study document.
2. LangChain document loaders extract the text.
3. Text is divided into smaller chunks.
4. Hugging Face embedding model converts text into vectors.
5. ChromaDB stores the embeddings and document content.
6. When a student asks a question, the retriever searches for relevant chunks using MMR.
7. Retrieved information is passed to the Groq LLM.
8. The LLM generates an answer based on the retrieved context.

## Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Study-Assistant
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure environment variables**

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

**6. Run the application**

```bash
streamlit run app.py
```

## Future Scope

- Develop a complete application using the MERN stack.
- Implement user authentication and individual student accounts.
- Allow students to upload and manage multiple documents.
- Maintain separate document collections for different users.
- Improve the RAG pipeline for better retrieval accuracy.

## Learning Outcomes

Through this project, I gained practical experience with:

- Retrieval-Augmented Generation (RAG).
- LangChain components and workflows.
- Text embeddings and vector databases.
- Semantic search and MMR retrieval.
- Integrating LLMs using Groq.
- Building document-based question-answering systems.

---
**Project Status:** Initial RAG implementation completed. Further development is planned.
