from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    """Context information for the weather agent."""
    user_id: str


@dataclass
class ResponseFormat:
    """Structured response format for weather agent replies."""
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("get_weather", description="Get the current weather for a given city", return_direct=False)
def get_weather(city: str):
    """Fetch current weather data for the given city."""
    weather_response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
    return weather_response.json()


@tool("locate_user", description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    """Look up a user's city based on the context."""
    match runtime.context.user_id:
        case "user_1":
            return "Nairobi"
        case "user_2":
            return "New York"
        case _:
            return "London"
        

model = ChatOllama(
    model="gemma4",
    temperature=0.7
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = "You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.",
    context_schema = Context,
    response_format = ResponseFormat,
    checkpointer = checkpointer
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like?"}
    ]},
    config = config,
    context = Context(user_id="user_1")
)

print(response)
