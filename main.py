from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2"
)

vectore_store = Chroma(
    collection_name="example_collection",
    persist_directory="./Chroma_DB",
    embedding_function=embedding
)

retriver = vectore_store.as_retriever(
    search_type = "mmr",
    search_kwargs={"k": 4, "lambda_mult":0.5}
)

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

#prompt 


prompt = PromptTemplate.from_template("""
You are a helpful AI study assistant. Answer the student's question using the provided context.

Context: {context}

Question: {question}

Instructions:
- Explain in simple, student-friendly language.
- Keep answers clear and relevant.
- Use examples if needed.
- Do not make up information.
- If the answer is not available in the context, say you couldn't find it in the uploaded document.

Answer:
""")

print("-------Rag System Created-------")
while True:
    para = input("Enter Question: ")

    docs = retriver.invoke(para)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context":context, "question":para})

    result = llm.invoke(final_prompt)
    print(result.content)