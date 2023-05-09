from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.chains.question_answering import load_qa_chain

# Constants
GPT4ALL_MODEL_PATH = "./models/gpt4all-converted.bin"
LLAMA_MODE_PATH = "./models/ggml-model-q4_0.bin"
DOC_TXT_PATH = "./docs/UVM.txt"
INDEX_DIRECTORY_PATH = "./UVM_INDEX1"

CONTEXT_SIZE = 2048
DOC_SPLIT_CHUNK = 512
DOC_CHUNK_OVERLAP = 32
CPU_THREADS = 12

INDEX_DOC: bool = True # set True for the first run in order to create doc indexes, then set it to False

def main():
    # load LLaMa embeddings for document preparation
    embeddings = LlamaCppEmbeddings(model_path=LLAMA_MODE_PATH)

    # index the document and save it
    if INDEX_DOC:
        with open(DOC_TXT_PATH) as f:
            document = f.read()
        
        # split the document
        text_splitter = CharacterTextSplitter(chunk_size=DOC_SPLIT_CHUNK, chunk_overlap=DOC_CHUNK_OVERLAP)
        texts = text_splitter.split_text(document)

        # index document splits
        docsearch = Chroma.from_texts(texts, embeddings,
                                    metadatas=[{"source": str(i)} for i in range(len(texts))],
                                    persist_directory=INDEX_DIRECTORY_PATH)
        # save indexes to drive
        docsearch.persist()

    # # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=INDEX_DIRECTORY_PATH, embedding_function=embeddings).as_retriever()

    # load GPT model
    llm = GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=CONTEXT_SIZE, verbose=True, n_threads=CPU_THREADS)
    chain = load_qa_chain(llm, chain_type="stuff")

    while True:
        query = input("\n\nQuestion: ")
        if query.lower() == 'exit':
            break

        print("Search relevant docs.")
        # get relevant document chunks
        docs = vectordb.get_relevant_documents(query)
        print("Done searching docs.")

        print("Waiting for answear...") 
        # retrieve an answear based on the given context ### HINT it takes a lot of time to generate an answear
        print(chain.run(input_documents=docs, question=query, return_only_outputs=True))


if __name__ == "__main__":
    main()