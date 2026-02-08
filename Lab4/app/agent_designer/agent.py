import os
from typing import List

from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL", "qwen/qwen3-32b")

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)

class AgentClarifier(BaseModel):
    clear: bool
    question: str

class AgentBlueprint(BaseModel):
    agent_name: str = Field(description="Name of the agent to be created")
    purpose: str = Field(description="What the agent is supposed to do")
    instructions: str = Field(description="System instructions for the agent")
    tools_needed: List[str] = Field(description="Names of tools the agent should have")

@ai_function(
    name="normalize_agent_name",
    description="Converts a free-text agent idea into a clean agent name"
)
def normalize_agent_name(idea: str) -> dict:
    name = idea.lower().replace(" ", "-")
    return {"normalized_name": f"{name}-agent"}

clarifier = ChatAgent(
    chat_client=client,
    name="clarifier-agent",
    description="Gets into the details of the user's idea",
    instruction="""
    You determine if the user's idea is clear enough to start designing.

    Rules:
    1) If the idea is clear enough return True
    2) If the idea is not clear enough return False and a single question for the user so he can clarify his idea.
    """,
    output_model=AgentClarifier,
)

agent = ChatAgent(
    chat_client=client,
    name="agent-builder-agent",
    description="Builds agent blueprints based on user requests",
    instructions="""
        You are an agent builder.

        Your task:
        - The user describes an agent they want.
        - Understand as much as possible how they want it.
        - You design that agent and return a structured blueprint.

        Rules:
        1) Always create a clear agent name.
        2) Clearly describe the agent's purpose.
        3) Write simple, understandable system instructions.
        4) Suggest 1–3 tools the agent should use.
        5) Use the normalize_agent_name tool to generate the agent name.
        6) Return ONLY structured output.
        7) Always check if the idea is clear enough using the clarifier tool.
        8) if clarifier returned False, ask a single question based on the returned value from clarifier.
        8) If the user asks anything unrelated to agent design, you will decline answering politely.
        9) Greeting and goodbyes are allowed in contrast to Rule no. 7.
    """,
    tools=[
        normalize_agent_name,
        clarifier.as_tool(),
    ],
    output_model=AgentBlueprint,
)
