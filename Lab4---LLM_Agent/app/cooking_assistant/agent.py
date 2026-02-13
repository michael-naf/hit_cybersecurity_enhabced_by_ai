import os
from typing import List, Any
from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel



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

#---------------------------------------------------#
#               Agent Data Structures               #
#---------------------------------------------------#

class Recipe(BaseModel):
    title: str | None = None
    ingredients: List[str]
    steps: List[str]

class ShoppingList(BaseModel):
    items: str | None = None

class CookingAssistantResponse(BaseModel):
    response_text: str
    current_recipe: Recipe | None = None
    shopping_list: ShoppingList | None = None

#-----------------------------------#
#               Tools               #
#-----------------------------------#

# Tool 1 for finding the recipes.
recipe_finder = ChatAgent(
    chat_client=client,
    name="recipe-finder-agent",
    description="Finds a recipe based on the user's desires",
    instructions="""
        Find a suitable recipe for the user based on his demands.

        Rules:
        - Make sure to ask the user about allergies and preferences before finding a recipe.
        - return the meal title, list of ingredients needed and list of steps for preparing the meal.
        - return ONLY if the user is satisfied with the recipe.

        IMPORTANT: You must provide a full Recipe object including a 'title', 'ingredients', and 'steps'. Do not leave the title blank.
    """,
    output_model=Recipe,
)



# Tool 2 for generating the shopping list.
@ai_function(
    name="shopping_list_generator",
    description="Compares recipe ingredients against what the user has to create a shopping list."
)
def shopping_list_generator(needed_ingredients: List[str], user_has: str) -> dict:
    """
    Calls the LLM to filter out ingredients the user already owns.
    """
    # Create a prompt for the LLM to filter ingredients
    prompt = f"""
    Recipe ingredients: {needed_ingredients}
    User already has: {user_has}
    
    Task: Create a comma-separated list of ONLY the items the user still needs to buy.
    Output format: "item1, item2, item3"
    If they have everything, return "None".
    """

    response = client.complete(prompt)
    return {"items": response.strip()}


# Tool 3: Final Formatter
@ai_function(
    name="output_formatting",
    description="Formats the final response for the user."
)
def output_format(response_text: str, recipe: Recipe = None, shopping_list: Any = None) -> dict:
    parsed_shopping_list = None
    if shopping_list:
        if isinstance(shopping_list, list):
            parsed_shopping_list = ShoppingList(items=", ".join(shopping_list))
        elif isinstance(shopping_list, dict):
            parsed_shopping_list = ShoppingList(**shopping_list)
        else:
            parsed_shopping_list = ShoppingList(items=str(shopping_list))
    
    
    response_obj = CookingAssistantResponse(
        response_text=response_text,
        current_recipe=recipe,
        shopping_list=parsed_shopping_list
        )
    return response_obj.model_dump()


#---------------------------------------------#
#           Cooking Assistant Agent           #
#---------------------------------------------#

agent = ChatAgent(
    chat_client=client,
    name="cooking-assistant",
    description="Main assistant for recipes and shopping.",
    instructions="""
    You are a Cooking Assistant.
    
    Workflow:
    1. if the user asks for a recipe Use 'recipe_finder' to help the user choose a meal.
    2. Once a recipe is chosen, ask the user what ingredients they already have.
    3. Use 'shopping_list_generator' to create their list.
    4. Use 'output_format' to present the final result to the user.
    
    Constraint: Only discuss cooking and food.
    """,
    tools=[
        shopping_list_generator,
        recipe_finder.as_tool(),
        output_format,
    ],
    output_model=CookingAssistantResponse,
)
