#load chunk embedding  vector-databse

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

data = PyPDFLoader(r"C:\Users\Admin\Desktop\RAG project\Document Loaders\DBMS Notes.pdf")

docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
    )

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="example_collection",
    embedding=embeddings,
    persist_directory="./Chroma_DB"
)
