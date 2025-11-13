from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def LoadVectorStore(persist_directory="db/chroma_db"):
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store


def main():
    persist_directory = "db/chroma_db"
    vector_store = LoadVectorStore(persist_directory)
    print("Vector store loaded successfully.")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # TESTING ONLY ----------- START
    sample_query = "The Vanguard Group percentage as shareholder?"
    relevant_docs = retriever.invoke(sample_query)

    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")


if __name__ == "__main__":
    main()