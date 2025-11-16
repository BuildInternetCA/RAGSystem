from retrieval import RetrivalEngine

def main():
    engine = RetrivalEngine()
    docs = engine.InvokeRetriever("What is The Vanguard Group percentage ?")
    for i, doc in enumerate(docs):
        print(f"\nRelevant Document {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:200]}...")
    print("Vector store is ready for use.")


if __name__ == "__main__":
    main()