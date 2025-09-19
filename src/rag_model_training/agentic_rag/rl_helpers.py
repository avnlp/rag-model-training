# This code is based on the implementation from: https://github.com/dCaples/AutoDidact/blob/main/rl_helpers.py.

from dataclasses import dataclass

import nest_asyncio
import torch
from trl.trainer.grpo_trainer import apply_chat_template

from .prompts import extract_json_objects, get_initial_chat
from .search_module import search

nest_asyncio.apply()


@dataclass
class AgenticOutputs:
    """Dataclass to store agent generation outputs."""

    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


def run_agent_generations(generate_fn, tokenizer, chat_states):
    """Run generation for chat states requiring assistant responses.

    Args:
        generate_fn: Function to generate responses
        tokenizer: Tokenizer for processing text
        chat_states: List of chat states

    Returns:
        list: Updated chat states
    """
    prompts = []
    batch_indices = []
    # Prepare prompts for chat states needing an assistant response.
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            continue

        if chat_state["messages"][-1]["role"] in ["ipython", "user"]:
            prompt = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
            prompts.append(prompt)
            batch_indices.append(idx)

    if prompts:
        responses = generate_fn(prompts)
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            full_response = responses[i].outputs[0].text if hasattr(responses[i], "outputs") else responses[i]
            # Handle both string responses and response objects
            if isinstance(full_response, str):
                assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            else:
                assistant_response = str(full_response).split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            chat_state["messages"].append({"role": "assistant", "content": assistant_response})
    return chat_states


def check_finished_chats(chat_states):
    """Check which chat states are finished (no more function calls).

    Args:
        chat_states: List of chat states

    Returns:
        list: Updated chat states with finished flag
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"
        assistant_response = chat_state["messages"][-1]["content"]
        function_calls = extract_json_objects(assistant_response)
        if len(function_calls) == 0:
            chat_state["finished"] = True
    return chat_states


def run_tool_calls(chat_states):
    """Execute tool calls found in chat states.

    Args:
        chat_states: List of chat states

    Returns:
        list: Updated chat states with tool call results
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant to run tool calls"
        try:
            assistant_response = chat_state["messages"][-1]["content"]
            function_calls = extract_json_objects(assistant_response)
            if len(function_calls) > 1:
                msg = "Expected only one function call in assistant response"
                raise ValueError(msg)
            elif len(function_calls) == 1:
                function_call = function_calls[0]
                query = function_call["function"]["parameters"]["query"]
                results = search(query, return_type=str, results=2)
                chat_state["messages"].append({"role": "ipython", "content": results})
        except Exception as e:
            chat_state["messages"].append({"role": "system", "content": f"Error during post-processing: {e!s}"})
            chat_state["finished"] = True
    return chat_states


def get_mask(text, tokenizer):
    """Create a mask for assistant responses in the conversation.

    Args:
        text: Conversation text
        tokenizer: Tokenizer for processing text

    Returns:
        torch.Tensor: Mask tensor with 1s for assistant response tokens, 0s elsewhere
    """
    encoding = tokenizer(text, add_special_tokens=False)
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_ranges = []
    i = 0
    while i < len(encoding.input_ids) - 1:
        if encoding.input_ids[i] == start_header_id and encoding.input_ids[i + 1] == assistant_token:
            i += 2
            while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
                i += 1
            i += 2
            start_idx = i
            while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                i += 1
            end_idx = i
            assistant_ranges.append((start_idx, end_idx))
        else:
            i += 1
    mask = [0] * len(encoding.input_ids)
    for start_idx, end_idx in assistant_ranges:
        for idx in range(start_idx, end_idx):
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.int)


def check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer):
    """Check if chat states have exceeded maximum new tokens limit.

    Args:
        chat_states: List of chat states
        max_new_tokens: Maximum allowed new tokens
        tokenizer: Tokenizer for processing text

    Returns:
        list: Updated chat states with finished flag for exceeded limits
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        initial_length = chat_state["initial_length"]
        new_length = get_chat_num_tokens(chat_state, tokenizer)
        if new_length - initial_length > max_new_tokens:
            chat_state["finished"] = True
    return chat_states


def get_chat_num_tokens(chat_state, tokenizer):
    """Get the number of tokens in a chat state.

    Args:
        chat_state: Chat state dictionary
        tokenizer: Tokenizer for processing text

    Returns:
        int: Number of tokens in the chat
    """
    chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
    return tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().shape[0]


def run_agent(generate_fn, tokenizer, questions, max_generations=5, max_new_tokens=4096):
    """Run the agent to completion for a batch of questions.

    Args:
        generate_fn: Function to generate model responses
        tokenizer: Tokenizer for processing text
        questions: List of questions to process
        max_generations: Maximum number of generation steps
        max_new_tokens: Maximum number of new tokens allowed

    Returns:
        AgenticOutputs: Agent outputs containing tokens, masks, and responses
    """
    chat_states = [get_initial_chat(q) for q in questions]
    # set the initial_prompt length
    for chat_state in chat_states:
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)

    # agent loop
    for _i in range(max_generations):
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_finished_chats(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer)

    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])

    def split_prompt_assistant(convo_text):
        """Split conversation text into prompt and assistant response parts."""
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            msg = "Could not find assistant marker in conversation text."
            raise ValueError(msg)
            return convo_text, ""
        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[: idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker) :]
        return prompt, assistant_response

    str_chats = [apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states]
    prompt_toks, response_toks, response_masks = [], [], []
    for str_chat in str_chats:
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze())
        response_toks.append(
            tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()[:max_new_tokens]
        )
        mask = get_mask(str_chat, tokenizer)[len(prompt_toks[-1]) :][:max_new_tokens]

        response_masks.append(mask)

    final_response_str = [chat["messages"][-1]["content"] for chat in chat_states]
    full_chat_states = chat_states
    agentic_outputs = AgenticOutputs(
        prompt_tokens=prompt_toks,
        response_tokens=response_toks,
        response_masks=response_masks,
        final_response_str=final_response_str,
        full_chat_states=full_chat_states,
    )

    return agentic_outputs
