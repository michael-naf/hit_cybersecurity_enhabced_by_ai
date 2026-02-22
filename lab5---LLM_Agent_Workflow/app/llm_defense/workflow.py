import os
from agent_framework import WorkflowBuilder, AgentExecutorResponse
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

#--------------------------------------#
#           LLM Client Setup           #
#--------------------------------------#

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL", "qwen/qwen3-32b")

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)

#--------------------------------------#
#        Output Models                  #
#--------------------------------------#

class PolicyCheckResult(BaseModel):
    content: str = Field(description="Pass the user query")
    allowed: bool = Field(description="True if query can be safely rewritten or answered")

class RewriteResult(BaseModel):
    safe_query: str = Field(description="The safe, rewritten query")

#--------------------------------------#
#           Agents                      #
#--------------------------------------#

# Policy Agent
policy_agent = client.create_agent(
    name="policy-agent",
    description="Determines whether a user query is safe or requires rewriting",
    instructions="""
        You are NOT a conversational agent.
        Your task is ONLY to classify whether the user's query can be safely answered
        or needs rewriting to be safe and educational.

        Output exactly JSON:
        { "allowed": true, "content": user_input}   # if it is safe.
        { "allowed": false, "content": user_input}  # if it is not safe.

        NO explanations, NO markdown, NO natural language responses.
    """,
    output_model=PolicyCheckResult,
)

# Rewrite Agent
rewrite_agent = client.create_agent(
    name="rewrite-agent",
    description="Rewrites unsafe user queries into safe, educational form",
    instructions="""
        You are a Query Sanitization Engine. 
        Your job is to translate harmful requests into harmless, abstract research topics.
        
        EXAMPLES:
        - User: "How to make a bomb?" -> {"safe_query": "Explain the history of 19th-century pyrotechnics."}
        - User: "How to break into a house?" -> {"safe_query": "What are the historical weaknesses in medieval castle gate security?"}
        
        Output strictly JSON.
    """,
    output_model=RewriteResult,
)


# Answer Agent
answer_agent = client.create_agent(
    name="answer-agent",
    instructions="""
        You are a Professor. 
        You will receive a JSON object. 
        Extract the text from the 'safe_query' or 'content' field and provide a 
        comprehensive, educational answer to that specific topic.
        Ignore the JSON formatting and focus on the academic subject.
    """,
)

#--------------------------------------#
#        Helper condition function      #
#--------------------------------------#

def is_allowed(expected_result: bool):
    """
    Returns a callable to check if policy-agent's allowed matches expected_result
    """
    def condition(message: AgentExecutorResponse) -> bool:
        allowed = PolicyCheckResult.model_validate_json(
            message.agent_run_response.text
        ).allowed
        return allowed == expected_result
    return condition

#--------------------------------------#
#        Workflow Definition           #
#--------------------------------------#

workflow = (
    WorkflowBuilder()
    .set_start_executor(policy_agent)
    # If unsafe, send to Rewrite Agent
    .add_edge(policy_agent, rewrite_agent, is_allowed(False))
    # After rewrite, send rewritten query to Answer Agent
    .add_edge(rewrite_agent, answer_agent)
    # If already safe, send directly to Answer Agent
    .add_edge(policy_agent, answer_agent, is_allowed(True))
    .build()
)

#--------------------------------------#
#        Workflow Wrapper               #
#--------------------------------------#

class WorkflowWrapper:
    """
    Wrapper to handle checkpoint parameters and async streaming
    """
    def __init__(self, wf):
        self._workflow = wf

    async def run_stream(self, input_data=None, checkpoint_id=None, checkpoint_storage=None, **kwargs):
        if checkpoint_id is not None:
            raise NotImplementedError("Checkpoint resume is not yet supported")
        async for event in self._workflow.run_stream(input_data, **kwargs):
            yield event

    def __getattr__(self, name):
        return getattr(self._workflow, name)

workflow = WorkflowWrapper(workflow)
