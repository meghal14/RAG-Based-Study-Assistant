# ------ Recursive Text Splitter -------

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = TextLoader(r"C:\Users\Admin\Desktop\RAG project\Document Loaders\test.txt")

docs = data.load()

# text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=15, chunk_overlap=1)
# texts = text_splitter1.split_documents(docs)

# for  i in texts:
#     print(i.page_content)
#     print()

#  ---- Character splitter (Chunking) ----
from langchain_text_splitters import CharacterTextSplitter

# text_splitter2 = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=10, chunk_overlap=1
# )
# texts2 = text_splitter2.split_documents(docs)

# for  i in texts2:
#     print(i.page_content)




from langchain_text_splitters import TokenTextSplitter

text_splitter3 = TokenTextSplitter(chunk_size=5, chunk_overlap=0)

texts3 = text_splitter3.split_documents(docs);

for i in texts3:
    print(i.page_content)
    print()
