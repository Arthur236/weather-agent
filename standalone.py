from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model = "claude-haiku-4-5",
    temperature = 0.1,
)

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming"),
    HumanMessage("Hello, what is Python?"),
    AIMessage("Python is a high-level, interpreted programming language known for its readability and versatility. It is widely used for web development, data analysis, artificial intelligence, scientific computing, and more. Python's syntax emphasizes code readability, making it an excellent choice for beginners and experienced developers alike."),
    HumanMessage("When was it released?")
]

for chunk in model.stream(conversation):
    print(chunk.text, end="", flush=True)
