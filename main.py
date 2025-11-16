from llmsystem import LLMEngine

def main():
    llm_engine = LLMEngine()
    query = "What is The Vanguard Group percentage?"
    k = 5

    # Get relevant documents using the retriever
    relevant_docs = llm_engine.GetRetriever(query, k)

    # Get the model with the prompt template
    chain = llm_engine.GetModelWithTemplate(relevant_docs, k)

    # Prepare the documents string
    documents_str = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    # Run the chain with the query and documents
    response = chain.invoke({"query" : query, "documents" :documents_str })

    print("Response:")
    print(response)


if __name__ == "__main__":
    main()