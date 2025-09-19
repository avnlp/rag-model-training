"""Prompt templates and utilities for the Agentic RAG model.

This module contains functions for building prompts for the Agentic RAG model,
including system prompts, user prompts, and utilities for processing responses.
"""

import json
import re
from datetime import datetime

# Tool definition for search corpus
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Search over the knowledge corpus with a given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the knowledge corpus with",
                },
            },
            "required": ["query"],
        },
    },
}


def get_system_prompt():
    """Get the system prompt with current date.

    Returns:
        str: The formatted system prompt including the current date.
    """
    current_date = datetime.now().strftime("%d %b %Y")
    return f"""Cutting Knowledge Date: December 2023
    Today Date: {current_date}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.
"""


def build_user_prompt(q):
    """Build a user prompt with the question and search tool definition.

    Args:
        q (str): The question to ask

    Returns:
        str: Formatted user prompt containing the search tool definition and question.
    """
    user_prompt = f"""You are a research assistant, and you use the search_corpus tool to find answers to questions.
    Given a question, answer it using by doing searches using the search_corpus tool.
    To use the search_corpus tool, respond with a JSON for a function call with its proper arguments.

    You may also reason in any message, thinking step by step about how to answer the question. Wrap your reasoning in <reasoning> and </reasoning> tags.

    {json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}

    Question: {q}
    """
    return user_prompt


def get_initial_chat(question):
    """Initialize a chat state with the question.

    Args:
        question (str): The question to ask

    Returns:
        dict: Initial chat state with system and user messages.
            Contains a list of message dictionaries with 'role' and 'content' keys.
    """
    return {
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": build_user_prompt(question)},
        ]
    }


def extract_json_objects(text):
    """Extracts JSON objects (dictionaries) from a text that may contain multiple JSON objects.

    Args:
        text (str): The input text possibly containing JSON objects.

    Returns:
        list: A list of parsed JSON objects (dictionaries) extracted from the text.
    """
    results = []
    length = len(text)
    i = 0

    while i < length:
        # Look for the start of a JSON object
        if text[i] == "{":
            start = i
            stack = 1
            i += 1
            # Continue until we find the matching closing brace
            while i < length and stack > 0:
                if text[i] == "{":
                    stack += 1
                elif text[i] == "}":
                    stack -= 1
                i += 1
            # Only attempt to decode if the braces are balanced
            if stack == 0:
                candidate = text[start:i]
                try:
                    obj = json.loads(candidate)
                    # Optionally, ensure it's a dictionary if that's what you expect
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    # If it's not valid JSON, skip it.
                    pass
        else:
            i += 1
    return results


def remove_reasoning(text: str) -> str:
    """Removes all content between <reasoning> and </reasoning> tags,
    including the tags themselves.

    Args:
        text (str): The input text that may contain <reasoning>...</reasoning> tags.

    Returns:
        str: The text with the tags and their content removed.
    """
    # The regex pattern matches from <reasoning> to </reasoning> non-greedily.
    pattern = r"<reasoning>.*?</reasoning>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text
