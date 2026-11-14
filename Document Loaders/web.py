from langchain_community.document_loaders import WebBaseLoader

url = "https://www.samsung.com/in/smartphones/galaxy-s26/?srsltid=AYCX9vL5ZFPYCYJ-z1pLn_wch2KKbTlegNqviPCqKU0h5titPIp6p2O8"

data = WebBaseLoader(url)
docs = data.load()

print(docs[0].page_content)
