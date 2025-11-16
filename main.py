from llmsystem import LLMEngine



def main():
    llm_engine = LLMEngine()
    
    query = "What is The Vanguard Group percentage?"
    k = 5

    while True:
        user_input = input("Enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        query = user_input
        response = llm_engine.GetModelWithTemplate(query, k)
        print("Response:")
        print(response)


if __name__ == "__main__":
    main()