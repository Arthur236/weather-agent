import time

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (AgentMiddleware, ModelRequest, ModelResponse,
                                         wrap_model_call)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()


class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0
    
    def before_agent(self, state: AgentState, runtime):
        """Hook called before the agent processes a request."""
        self.start_time = time.time()
        print("Before agent processing...")
    
    def before_model(self, state: AgentState, runtime):
        """Hook called before the model processes a request."""
        print("Before model processing...")
        
    def after_model(self, state: AgentState, runtime):
        """Hook called after the model processes a request."""
        print("After model processing...")
        
    def after_agent(self, state: AgentState, runtime):
        """Hook called after the agent processes a request."""
        print("After agent processing...", time.time() - self.start_time)

model = ChatOllama(
    model="gemma4",
)

agent = create_agent(
    model=model,
    middleware=[HooksDemo()],
)

response = agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 1 + 1?"),
    ]
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
