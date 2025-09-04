from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_qa(pdf_path, deepseek_api_key):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # DeepSeek LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1"
    )

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
