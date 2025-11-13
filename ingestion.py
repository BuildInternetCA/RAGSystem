import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def LoadDocuments(directory_path):
    loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # TESTING ONLY ----------- START
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
    # TESTING ONLY ----------- END
 
    return documents


def SplitDocuments(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    split_docs = text_splitter.split_documents(documents)

    # TESTING ONLY ----------- START
    for i, doc in enumerate(split_docs[:2]):  # Show first 2 split documents
        print(f"\nSplit Document {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
    # TESTING ONLY ----------- END

    return split_docs

def CreateVectorStore(documents, persist_directory="db/chroma_db"):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vector_store

def main():
    directory_path = "docs"
    persist_directory = "db/chroma_db"
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        print("Vector store already exists. Loading existing store...")
        embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vector_store
    # STEP 1: Load Documents
    print("Loading documents...")
    documents = LoadDocuments(directory_path)
    # STEP 2: Split Documents
    print("Splitting documents...")
    split_documents = SplitDocuments(documents)
    # STEP 3: Create Vector Store
    print("Creating vector store...")
    vector_store = CreateVectorStore(split_documents, persist_directory)
    print("Vector store created and persisted.")
    return vector_store

if __name__ == "__main__":
    main()