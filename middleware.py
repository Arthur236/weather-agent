from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_ollama import ChatOllama

load_dotenv()


@dataclass
class Context:
    """Context schema for runtime request handling."""
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Return a role-specific prompt based on the request context."""
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case "expert":
            return base_prompt + " Provide detailed technical explanations."
        case "beginner":
            return base_prompt + " Keep your explanations simple and easy to understand."
        case "child":
            return base_prompt + " Explain everything like I'm five years old."
        case _:
            return base_prompt


model = ChatOllama(
    model="gemma4",
)

agent = create_agent(
    model=model,
    middleware=[user_role_prompt],
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain PCA."}]},
    context=Context(user_role="beginner")
)

print(response)
