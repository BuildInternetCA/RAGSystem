from retrieval import RetrivalEngine
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class LLMEngine:
    def __init__(self):
        pass

    def GetRetriever(self, query, k=5):
        retriever = RetrivalEngine().InvokeRetriever(query, k)
        return retriever

    def GetModelWithTemplate(self, relevant_docs, k=5):
        #relevant_docs = self.GetRetriever(query, k)
        #{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
        combined_input = """Based on the following documents, please answer this question: {query}
        Documents:
        {documents}
        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
        """

        model = OllamaLLM(model="llama3.2")

        prompt = ChatPromptTemplate.from_template(combined_input)
        
        chain = prompt | model

        return chain