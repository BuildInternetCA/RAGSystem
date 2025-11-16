from retrieval import RetrivalEngine
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class LLMEngine:
    def __init__(self):
        self.chatHistory = []
        self.model = OllamaLLM(model="llama3.2")
        pass

    def GetRetriever(self, query, k=5):
        self.chatHistory
        retriever = RetrivalEngine().InvokeRetriever(query, k)
        return retriever

    def GetModelWithTemplate(self, queryPlain, k=5):
        query = self.RewriteQuery(queryPlain)
        print(f"Rewritten Query: {query}")
        relevant_docs = self.GetRetriever(query, k)
        #{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
        combined_input = f"""Based on the following documents, please answer this question: {query}
        Documents:
        {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
        """
        # prompt = ChatPromptTemplate.from_template(combined_input)
        # chain = prompt | self.model
        # return chain
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
        ] + self.chatHistory + [
            HumanMessage(content=combined_input)
        ]
        results = self.model.invoke(messages)
        answer = results
        self.AddToChatHistory("user", query)
        self.AddToChatHistory("ai", answer) 
        return answer
    
    def AddToChatHistory(self, role, content):
        if role == "user":
            self.chatHistory.append(HumanMessage(content=content))
        elif role == "ai":
            self.chatHistory.append(AIMessage(content=content))
        else:
            self.chatHistory.append(SystemMessage(content=content))
        
    def RewriteQuery(self, query):
        if self.chatHistory == []:
            return query
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + self.chatHistory + [
            HumanMessage(content=f"New question: {query}")
        ]
        result = self.model.invoke(messages)
        search_question = result
        return search_question