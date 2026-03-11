# 🍳 Cooking Assistant

The Cooking Assistant is an AI agent designed to guide users through the complete cooking workflow, from recipe discovery to shopping list preparation. Its purpose is to provide a structured, user-friendly assistant that helps people plan meals efficiently while considering their preferences and available ingredients.

## 📝 Agent Purpose

The agent’s system prompt is configured to act as a technical cooking assistant. Its responsibilities include:

**1.** Assisting the user in finding recipes that match their dietary preferences, allergies, and desired cuisine.

**2.** Querying the user for ingredients they already possess.

**3.** Generating a shopping list for missing ingredients.

**4.** Presenting a fully formatted, structured response including the recipe and shopping list.

The agent’s role is to orchestrate multiple tools, ensuring a smooth and safe workflow strictly focused on cooking and food.

## 🛠️ Agent Tools

The Cooking Assistant leverages three primary tools:

### 1️⃣ recipe_finder

* This tool locates recipes based on the user’s preferences and dietary restrictions.

* Input: User preferences, dietary restrictions, or allergies.

* Output: A Recipe object containing the title of the dish, a list of ingredients, and step-by-step steps to prepare it.

### 2️⃣ shopping_list_generator

* Compares the required recipe ingredients against what the user already has.

* Input: A list of needed_ingredients and a string describing what the user already owns.

* Output: A dictionary containing items as a comma-separated string of ingredients the user still needs. Returns "None" if all ingredients are already available.

### 3️⃣ output_format

* Formats the final response for the user in a structured way.

* Input: response_text for user-facing messages, optional recipe object, and optional shopping_list.

* Output: A CookingAssistantResponse object with response_text, the current_recipe, and a shopping_list.

## 💬 Example Interaction

**User:** "I want to cook chicken tonight. Can you suggest a recipe?"

### Agent Workflow:

**1.** The assistant uses recipe_finder to suggest a recipe and asks about allergies and preferences.

**2.** Once a recipe is confirmed, the assistant asks what ingredients the user already has.

**3.** Using shopping_list_generator, it identifies missing ingredients.

**4.** Finally, output_format is used to generate a structured response.

**User:** "I already have chicken, garlic, and olive oil."

**Cooking Assistant Response:**
---
    {
        "response_text": "Here's your recipe and shopping list!",
        "current_recipe": {
            "title": "Garlic Chicken Stir-Fry",
            "ingredients": ["chicken", "garlic", "onions", "bell peppers", "olive oil", "spices"],
            "steps": ["Step 1: Chop ingredients...", "Step 2: Heat oil...", "Step 3: Cook chicken..."]
        },
        "shopping_list": {
            "items": "onions, bell peppers, spices"
        }
    }
---
The assistant ensures the user has a complete cooking plan, including both the recipe and a clear shopping list.