import os
import argparse
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DATA_DIR = "./patient_data"
CHROMA_DB_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 3 
MEDICAL_PROMPT = """You are a highly accurate medical AI assistant for a hospital reviewing patient records.
Answer the doctor's question using ONLY the retrieved patient records below.
If you don't know the answer or if the information is not in the context,
reply exactly: 'Information not found in patient records.' DO NOT hallucinate or guess.

Retrieved patient records:
{context}

Question: {input}"""


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def ingest_data():
    print("1. Loading patient documents...")
    docs = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader).load()
    print(f"   Loaded {len(docs)} documents.")

    print("2. Chunking documents...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    ).split_documents(docs)
    print(f"   Split into {len(chunks)} chunks.")

    print("3. Saving to Chroma...")
    db = Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_DB_DIR)
    db.persist()
    print("   Done!")


def query_system(question):
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Add GOOGLE_API_KEY to your .env file.")
        return

    print(f"\nQuestion: {question}\n")

    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embeddings())
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("human", MEDICAL_PROMPT)])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    print("Searching records and generating answer...")
    response = rag_chain.invoke({"input": question})

    print("\n=== AI RESPONSE ===")
    print(response["answer"])
    print("===================\n")

    print("Sources used:")
    for doc in response["context"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hospital AI RAG System")
    parser.add_argument("--ingest", action="store_true", help="Load patient files into the database")
    parser.add_argument("--query", type=str, help="Ask a question")
    args = parser.parse_args()

    if args.ingest:
        ingest_data()
    elif args.query:
        query_system(args.query)
    else:
        print("Usage:")
        print("  python hospital_rag.py --ingest")
        print('  python hospital_rag.py --query "What is John Doe\'s blood pressure?"')