import os

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from pydantic import BaseModel
from typing import Literal

class QuestionCheckResult(BaseModel):
    is_eligible: bool
    category: str  # "greeting", "weather", "geography", "other"
    reason: str

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL", "qwen/qwen3-32b")

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)

question_check_agent = ChatAgent(
    chat_client=client,
    name="question-check-agent",
    description="Classifies whether a user message can be answered",
    instructions="""
        You are NOT a conversational agent.

        Classify the user's message into exactly one intent:
        - greeting
        - goodbye
        - weather
        - geography
        - other

        Rules:
        - greeting and goodbye are always eligible
        - weather and geography are eligible
        - everything else is not eligible

        Return ONLY structured output. No explanations.
    """,
    output_model=QuestionCheckResult,
)

agent = ChatAgent(
    chat_client=client,
    name="geography-weather-agent",
    instructions="""
        You're a geography and weather assistant.
        You can elaborate just on topics allowed by question-check-agent, plus you 
        can greet and say goodbye to users.
        If the user message is not related to greetings or goodbyes, so call question-check-agent
        to determine whether the topic is allowed.
          If the topic is allowed, provide a detailed and informative answer. 
        If the topic is not allowed, 
          politely refuse to answer mentioning that you're a geography and weather assistant and
          cannot server requests other than allowed by your control mechanism.
    """,
    tools=[
        question_check_agent.as_tool(),
    ],
)
