from langchain_community.document_loaders import TextLoader

data = TextLoader(r"RAG project\Document Loaders\notes.txt")

docs = data.load()

print(docs[0].page_content)