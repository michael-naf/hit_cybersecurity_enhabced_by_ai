import os

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL", "qwen/qwen3-32b")

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)

agent = ChatAgent(
    chat_client=client,
    name="hello-world-agent",
    instructions="""
        You're a friendly agent.
        Ask the user for their name and greet them personally.
    """
)
