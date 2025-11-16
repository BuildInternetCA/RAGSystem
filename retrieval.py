from ingestion import IngestionEngine


class RetrivalEngine:
    def __init__(self):
        self.vector_store = IngestionEngine().GetDatabase()

    def GetRetriever(self, k=5):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    
    def InvokeRetriever(self, query, k=5):
        retriever = self.GetRetriever(k)
        relevant_docs = retriever.invoke(query)
        return relevant_docs