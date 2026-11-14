from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"C:\Users\Admin\Desktop\RAG project\Document Loaders\DBMS Notes.pdf")

docs = data.load()

print(len(docs))