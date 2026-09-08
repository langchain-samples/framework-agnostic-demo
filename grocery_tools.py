"""Shared in-store grocery shopping tools used by all three agent implementations.

These are plain Python functions so they can be reused across Google ADK, the
OpenAI Agents SDK, and the bare-code (no framework) implementation.
"""

# A tiny in-memory "store" so the demo works without any backend.
_STORE = {
    "milk": {"aisle": "D1 (Dairy)", "price": 3.49, "stock": 42},
    "eggs": {"aisle": "D2 (Dairy)", "price": 4.19, "stock": 30},
    "bread": {"aisle": "B3 (Bakery)", "price": 2.99, "stock": 18},
    "bananas": {"aisle": "P1 (Produce)", "price": 0.59, "stock": 120},
    "tomatoes": {"aisle": "P2 (Produce)", "price": 1.29, "stock": 64},
    "chicken breast": {"aisle": "M1 (Meat)", "price": 6.99, "stock": 25},
    "pasta": {"aisle": "A5 (Dry Goods)", "price": 1.79, "stock": 80},
    "olive oil": {"aisle": "A6 (Dry Goods)", "price": 8.49, "stock": 12},
    "cheddar cheese": {"aisle": "D3 (Dairy)", "price": 5.29, "stock": 22},
    "coffee": {"aisle": "A8 (Beverages)", "price": 9.99, "stock": 15},
}

_RECIPES = {
    "pasta": ["pasta", "tomatoes", "olive oil", "cheddar cheese"],
    "breakfast": ["eggs", "bread", "milk", "bananas"],
    "chicken dinner": ["chicken breast", "olive oil", "tomatoes", "pasta"],
}


def find_item_aisle(item: str) -> dict:
    """Find which aisle a grocery item is located in.

    Args:
        item: The name of the grocery item to locate (e.g. "milk").
    """
    key = item.strip().lower()
    if key in _STORE:
        return {"item": key, "aisle": _STORE[key]["aisle"]}
    return {"item": key, "aisle": None, "message": f"'{item}' not found in store map."}


def check_price(item: str) -> dict:
    """Check the current price of a grocery item.

    Args:
        item: The name of the grocery item to price (e.g. "eggs").
    """
    key = item.strip().lower()
    if key in _STORE:
        return {"item": key, "price_usd": _STORE[key]["price"]}
    return {"item": key, "price_usd": None, "message": f"'{item}' not found."}


def check_stock(item: str) -> dict:
    """Check how many units of an item are currently in stock.

    Args:
        item: The name of the grocery item to check (e.g. "bread").
    """
    key = item.strip().lower()
    if key in _STORE:
        qty = _STORE[key]["stock"]
        return {"item": key, "in_stock": qty > 0, "units": qty}
    return {"item": key, "in_stock": False, "units": 0, "message": f"'{item}' not found."}


def suggest_recipe(meal: str) -> dict:
    """Suggest a shopping list of ingredients for a meal idea.

    Args:
        meal: A meal or dish idea (e.g. "pasta", "breakfast", "chicken dinner").
    """
    key = meal.strip().lower()
    if key in _RECIPES:
        return {"meal": key, "ingredients": _RECIPES[key]}
    return {
        "meal": key,
        "ingredients": [],
        "message": f"No preset recipe for '{meal}'. Try: {', '.join(_RECIPES)}.",
    }


# Convenience collections for the different frameworks.
ALL_TOOLS = [find_item_aisle, check_price, check_stock, suggest_recipe]

SYSTEM_PROMPT = (
    "You are a friendly in-store shopping assistant for a grocery store. "
    "Help shoppers find items, check prices and stock, and plan meals using the "
    "available tools. Be concise and practical, and mention aisle numbers when "
    "you know them. If an item isn't in the store map, say so honestly."
)
