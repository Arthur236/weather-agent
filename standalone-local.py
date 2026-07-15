from dotenv import load_dotenv

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(
    model="gemma4",
    temperature=0.7
)

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming"),
    HumanMessage("Hello, what is Python?"),
    AIMessage("Python is a high-level, interpreted programming language known for its readability and versatility. It is widely used for web development, data analysis, artificial intelligence, scientific computing, and more. Python's syntax emphasizes code readability, making it an excellent choice for beginners and experienced developers alike."),
    HumanMessage("When was it released?")
]

for chunk in model.stream(conversation):
    print(chunk.text, end="", flush=True)
