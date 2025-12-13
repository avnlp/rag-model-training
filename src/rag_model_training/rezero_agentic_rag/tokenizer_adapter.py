# This code is based on the implementation from: https://github.com/menloresearch/ReZero/blob/main/src/tokenizer_adapter.py.
"""Tokenizer adapter implementations for different models.
This module provides adapter classes for handling different tokenizer formats.
"""

from abc import ABC, abstractmethod

import torch

from .config import logger


class TokenizerAdapter(ABC):
    """Base class for tokenizer adapters."""

    @abstractmethod
    def get_assistant_marker(self) -> str:
        """Get the assistant marker for the model."""
        pass

    @abstractmethod
    def get_end_marker(self) -> str:
        """Get the end marker for the model."""
        pass

    @abstractmethod
    def get_mask(self, text: str, tokenizer) -> torch.Tensor:
        """Get the mask for the model's response."""
        pass

    @abstractmethod
    def split_prompt_assistant(self, text: str) -> tuple[str, str]:
        """Split conversation text into prompt and assistant response."""
        pass


class LlamaTokenizerAdapter(TokenizerAdapter):
    """Adapter for Llama model tokenizer."""

    def get_assistant_marker(self) -> str:
        """Get the assistant marker."""
        return "<|start_header_id|>assistant<|end_header_id|>"

    def get_end_marker(self) -> str:
        """Get the end marker."""
        return "<|eot_id|>"

    def split_prompt_assistant(self, convo_text: str) -> tuple[str, str]:
        """Split the text into prompt and assistant parts.

        Args:
            convo_text: The text to split

        Returns:
            A tuple of (prompt, assistant)
        """
        # EXACT replication from rl_helpers.py but using existing method
        marker = self.get_assistant_marker()  # Use existing method but same value
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

    def get_mask(self, text: str, tokenizer) -> torch.Tensor:
        """Get the mask for the text.

        Args:
            text: The text to get the mask for
            tokenizer: The tokenizer to use

        Returns:
            A tensor of 0s and 1s where 1s indicate assistant tokens
        """
        # Log input
        logger.debug(f"Full text length: {len(text)}")

        # EXACT replication from rl_helpers.py but using existing methods
        encoding = tokenizer(text, add_special_tokens=False)
        # Use existing methods but same values
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        assistant_token = tokenizer.convert_tokens_to_ids("assistant")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = tokenizer.convert_tokens_to_ids(self.get_end_marker())  # Use existing method but same value

        # Log token IDs
        logger.debug(f"Tokenized length: {len(encoding.input_ids)}")
        logger.debug(f"Input IDs: {encoding.input_ids}")
        logger.debug(
            f"Special token IDs: start={start_header_id}, assistant={assistant_token}, end={end_header_id}, eot={eot_id}"
        )

        assistant_ranges = []
        i = 0
        while i < len(encoding.input_ids) - 1:
            if encoding.input_ids[i] == start_header_id and encoding.input_ids[i + 1] == assistant_token:
                logger.debug(f"Found assistant marker at position {i}")
                logger.debug(f"Assistant marker tokens: {encoding.input_ids[i : i + 2]}")
                i += 2
                while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
                    i += 1
                i += 2
                start_idx = i
                logger.debug(f"Found start of response at {start_idx}")
                logger.debug(f"Start token ID: {encoding.input_ids[start_idx]}")
                while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                    i += 1
                end_idx = i
                logger.debug(f"Found end of response at {end_idx}")
                logger.debug(f"End token ID: {encoding.input_ids[end_idx]}")
                logger.debug(f"Response token IDs: {encoding.input_ids[start_idx:end_idx]}")
                assistant_ranges.append((start_idx, end_idx))
            else:
                i += 1

        mask = [0] * len(encoding.input_ids)
        for start_idx, end_idx in assistant_ranges:
            for idx in range(start_idx, end_idx):
                mask[idx] = 1

        mask = torch.tensor(mask, dtype=torch.int)

        # Log final mask
        logger.debug(f"Final mask shape: {mask.shape}")
        logger.debug(f"Mask sum: {mask.sum().item()}")
        logger.debug(f"Mask: {mask}")

        # Additional debug info
        try:
            prompt, response = self.split_prompt_assistant(text)
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            response_tokens = tokenizer(response, add_special_tokens=False).input_ids

            logger.debug(f"Prompt length: {len(prompt)}")
            logger.debug(f"Response length: {len(response)}")
            logger.debug(f"Prompt token IDs: {prompt_tokens}")
            logger.debug(f"Response token IDs: {response_tokens}")
            logger.debug(f"Prompt: {prompt[:100]}...")
            logger.debug(f"Response: {response[:100]}...")
            logger.debug(f"Full input IDs length: {len(encoding.input_ids)}")
            logger.debug(f"Prompt + Response token IDs length: {len(prompt_tokens) + len(response_tokens)}")
            logger.debug(
                f"Difference in lengths: {len(encoding.input_ids) - (len(prompt_tokens) + len(response_tokens))}"
            )
        except Exception as e:
            logger.error(f"Error splitting prompt/response: {e}")

        return mask
