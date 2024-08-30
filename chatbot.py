from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Create a TextLoader object
loader = TextLoader("dataset/ai_job_market_insights_mini.csv")

# Create an OllamaEmbeddings object
embeddings = OllamaEmbeddings(model="llama3.1")

# Create a VectorstoreIndexCreator object
index_creator = VectorstoreIndexCreator(embedding=embeddings)

# Call from_loaders method
index = index_creator.from_loaders([loader])
print("indexing document in vector store completed!")

# Create a ChatOllama object
chat_llama3 = ChatOllama(model="llama3.1", temperature=0.7)

prompt = ""
while prompt.lower() != "exit":
    # Use ChatOllama object to answer questions
    prompt = input("Enter your query: ")
    answer = index.query(prompt, llm=chat_llama3)
    print("Llama3 Chatbot: " + answer)
