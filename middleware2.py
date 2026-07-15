from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelRequest, ModelResponse,
                                         wrap_model_call)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()


basic_model = ChatOllama(
    model="gemma4",
)

advanced_model = init_chat_model(
    model="claude-sonnet-5",
)


@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler) -> ModelResponse:
    """Select a model based on the request context."""
    message_count = len(request.messages)

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model

    return handler(request)


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selector],
)

response = agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 1 + 1?"),
    ]
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
