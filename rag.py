from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = OllamaEmbeddings(model="embeddinggemma")

texts = [
  "I love apples",
  "I enjoy oranges",
  "I think pears taste very good",
  "I hate bananas",
  "I dislike raspberries",
  "I love mangos",
  "I love Linux",
  "I hate Windows"
]

vector_store = InMemoryVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
)

# print(vector_store.similarity_search("What fruits does the person like?", k=3))
# print(vector_store.similarity_search("What fruits does the person hate?", k=3))

retriever = vector_store.as_retriever(search_kwargs={"k": 6})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="kb_search",
    description=(
        "Search the knowledge base of the user's own stated personal preferences, "
        "likes, and dislikes about fruits and other topics. Use this whenever the user "
        "asks what they like or dislike."
    ),
)

model = ChatOllama(
    model="gemma4",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant with access to a knowledge base tool called kb_search. For any question, you must search the knowledge base first using the kb_search tool to retrieve relevant information, then answer based on what you found."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What 3 fruits do I like and what 3 fruits do I dislike?"}
    ]
})

print(response)
