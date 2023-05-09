"""
Similar with qa_over_docs.py but with chat history retention and uses FAISS instead of Chroma
"""

from typing import List
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain


# Constants
GPT4ALL_MODEL_PATH = "./models/gpt4all-converted.bin"
LLAMA_MODE_PATH = "./models/ggml-model-q4_0.bin"
DOC_TXT_PATH = "./docs/UVM.txt"
INDEX_DIRECTORY_PATH = "./UVM_INDEX2"

CONTEXT_SIZE = 2048
DOC_SPLIT_CHUNK = 512
DOC_CHUNK_OVERLAP = 32
CPU_THREADS = 12

INDEX_DOC: bool = False # set True for the first run in order to create doc indexes, then set it to False


# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=LLAMA_MODE_PATH)

def load_documents() -> List:
    loader = TextLoader(DOC_TXT_PATH)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=DOC_SPLIT_CHUNK, chunk_overlap=DOC_CHUNK_OVERLAP)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)



def main():
    # load LLaMa embeddings for document preparation
    embeddings = LlamaCppEmbeddings(model_path=LLAMA_MODE_PATH)

    # index the document and save it
    if INDEX_DOC:
        sources = load_documents()
        chunks = split_chunks(sources)
        vectorstore = generate_index(chunks, embeddings)
        vectorstore.save_local(INDEX_DIRECTORY_PATH)

    # load GPT model
    llm = GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=CONTEXT_SIZE, verbose=True, n_threads=CPU_THREADS)
    index = FAISS.load_local(INDEX_DIRECTORY_PATH, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=CONTEXT_SIZE, return_source_documents=True)

    # Chatbot loop
    chat_history = []
    while True:
        query = input("\n\nQuestion: ")
        
        if query.lower() == 'exit':
            break
        ### HINT it takes a lot of time to generate an answear
        result = chain({"question": query, "chat_history": chat_history})

        print("Answer:", result['answer'])

        print("\n\nSource documents:", result['source_documents'])


if __name__ == "__main__":
    main()