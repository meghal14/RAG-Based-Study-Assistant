from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_core.documents import Document
docs = [
    Document(
        page_content="Python is a high-level programming language used in web development, automation, Artificial Intelligence, and Machine Learning.",
        metadata={"source": "python_notes.txt", "topic": "Python"}
    ),

    Document(
        page_content="Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions without being explicitly programmed.",
        metadata={"source": "ml_notes.txt", "topic": "Machine Learning"}
    ),

    Document(
        page_content="Retrieval Augmented Generation (RAG) combines information retrieval with Large Language Models. It retrieves relevant documents and uses them to generate accurate answers.",
        metadata={"source": "rag_notes.txt", "topic": "RAG"}
    )
]


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="Chroma_DB"
)

result = vectorstore.similarity_search("What is a branch of Artificial Intelligence?", k=2)

for r in result:
    print(r.page_content)
    print()