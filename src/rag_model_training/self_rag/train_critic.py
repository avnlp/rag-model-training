"""Self-RAG Critic Model Training.

Train a Critic model for Self-RAG with specialized tokens for retrieval and evaluation.

This implementation is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/train_special_tokens.py

This module implements the training pipeline for a Critic model in the Self-RAG framework.
The Critic model is responsible for evaluating and scoring the quality of retrieved documents
and generated responses. It uses special tokens to represent different aspects of the evaluation,
such as retrieval decisions, relevance, utility scores, and factuality.

Key Features:
- Adds special tokens for Self-RAG operations (e.g., [Retrieval], [Relevant], [Utility:N])
- Handles instruction-following data with proper formatting
- Supports efficient training with gradient accumulation and mixed precision
- Includes utilities for data loading, tokenization, and model training

Modifications from the original implementation:
1. Adapted for the Earnings Call Dataset format
2. Updated to work with the latest HuggingFace Transformers API
3. Enhanced configuration management
4. Explicit token handling and masking logic
"""

import json
import logging
from dataclasses import dataclass, field

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# Default configuration file path
DEFAULT_CONFIG_PATH = "train_critic_config.yaml"

# Special token to indicate ignored indices in the loss calculation
IGNORE_INDEX = -100

# Special tokens used in Self-RAG for retrieval and evaluation
SELF_RAG_TOKENS = [
    "[No Retrieval]",
    "[Retrieval]",
    "[Continue to Use Evidence]",
    "[Irrelevant]",
    "[Relevant]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]

# Template for formatting instructions and inputs
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "\n### Instruction:\n{instruction}\n"
    "\n### Input:\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)


@dataclass
class Config:
    """Configuration class for training the Critic model.

    Attributes:
        model_name: Name or path of the pre-trained language model to use.
        max_length: Maximum sequence length for tokenization.
        data_path: Path to the training data file (JSON format).
        output_dir: Directory to save the trained model and checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU/TPU core/CPU for training.
        gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass.
        learning_rate: The initial learning rate for AdamW optimizer.
        warmup_steps: Number of steps for the warmup phase of learning rate scheduler.
        logging_steps: Log every X updates steps.
        save_steps: Save checkpoint every X updates steps.
        use_special_tokens: Whether to add Self-RAG special tokens to the tokenizer.
        mask_special_tokens: Whether to mask special tokens during loss calculation.
    """

    # Model configuration
    model_name: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Name or path of the pre-trained model"},
    )
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})

    # Data configuration
    data_path: str = field(default="data/critic_data.json", metadata={"help": "Path to training data"})

    # Training configuration
    output_dir: str = field(
        default="./critic_model",
        metadata={"help": "Output directory for model checkpoints"},
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per device"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Number of steps for gradient accumulation"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate"})
    warmup_steps: int = field(default=100, metadata={"help": "Number of warmup steps"})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps"})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps"})

    # Special tokens configuration
    use_special_tokens: bool = field(default=True, metadata={"help": "Whether to add Self-RAG special tokens"})
    mask_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether to mask special tokens in loss calculation"},
    )


class CriticDataset(Dataset):
    """Dataset for training the Self-RAG Critic model.

    This dataset handles loading and preprocessing of training data for the Critic model.
    It formats the input-output pairs and handles special token masking for training.

    Args:
        data_path: Path to the JSON file containing training examples.
        tokenizer: Pre-trained tokenizer for text processing.
        config: Configuration object containing model and training parameters.
    """

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, config: Config):
        """Initialize the dataset with data and configuration.

        Args:
            data_path: Path to the JSON file containing training examples.
            tokenizer: Pre-trained tokenizer for text processing.
            config: Configuration object with model and training parameters.
        """
        self.tokenizer = tokenizer
        self.config = config

        # Load and validate training data
        try:
            with open(data_path, encoding="utf-8") as f:
                self.data = json.load(f)
            if not isinstance(self.data, list):
                msg = "Training data should be a list of examples"
                raise ValueError(msg)
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {e!s}")
            raise

        # Initialize special token IDs for masking during training
        self.special_token_ids = []
        if config.use_special_tokens and config.mask_special_tokens:
            self.special_token_ids = [
                tokenizer.convert_tokens_to_ids(tok)
                for tok in SELF_RAG_TOKENS
                if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
            ]

        logging.info(f"Loaded {len(self.data)} training examples")

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Returns:
            Number of examples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training example by index.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: Token IDs for the input sequence
                - labels: Token IDs for the target sequence (with IGNORE_INDEX for prompt tokens)
                - attention_mask: Attention mask for the input sequence
        """
        item = self.data[idx]

        # Format the instruction and input into a prompt
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()

        # Handle cases with and without input
        if inp:
            prompt = PROMPT_TEMPLATE.format(instruction=instr, input=inp)
        else:
            # Remove empty input section if no input is provided
            prompt = PROMPT_TEMPLATE.format(instruction=instr, input="").replace("### Input:\n\n", "")

        # Combine prompt with output and add end-of-turn token
        full_text = prompt + item.get("output", "") + "<|eot_id>"

        # Tokenize the full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.squeeze()
        labels = input_ids.clone()

        # Calculate prompt length in tokens and mask the prompt in labels
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Mask prompt tokens in labels to ignore them in loss calculation
        labels[: prompt_ids.size(0)] = IGNORE_INDEX

        # Mask special tokens if configured
        if self.config.mask_special_tokens:
            for i, tid in enumerate(input_ids.tolist()):
                if tid in self.special_token_ids:
                    labels[i] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),  # All tokens are attended to
        }


def setup_model_and_tokenizer(
    config: Config,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Initialize the model and tokenizer with the given configuration.

    This function:
    1. Loads the tokenizer and configures it with special tokens
    2. Loads the pre-trained language model
    3. Resizes the token embeddings if special tokens are added
    4. Initializes new token embeddings with the average of existing ones

    Args:
        config: Configuration object containing model parameters.

    Returns:
        A tuple containing:
            - Initialized language model
            - Configured tokenizer
    """
    # Initialize tokenizer with appropriate settings
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_0|>"

    # Add Self-RAG special tokens if enabled
    if config.use_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": SELF_RAG_TOKENS})

    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Resize token embeddings if special tokens were added
    if config.use_special_tokens:
        original_vocab_size = len(tokenizer) - len(SELF_RAG_TOKENS)
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new token embeddings with the average of existing ones
        with torch.no_grad():
            embeddings = model.get_input_embeddings()
            avg_embedding = embeddings.weight[:original_vocab_size].mean(dim=0)

            # Set new token embeddings to the average
            for i in range(original_vocab_size, len(tokenizer)):
                embeddings.weight[i] = avg_embedding.clone()

    return model, tokenizer


def pad_sequence(sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    """Pad sequences to the maximum length in the batch."""
    # Find maximum sequence length in the batch
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequence = torch.stack(
        [
            torch.cat(
                [
                    seq,
                    torch.full(
                        (max_len - seq.size(0),),
                        padding_value,
                        dtype=seq.dtype,
                        device=seq.device,
                    ),
                ]
            )
            for seq in sequences
        ]
    )
    return padded_sequence


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for the DataLoader to handle variable-length sequences.

    This function takes a batch of examples and pads them to the maximum sequence length
    in the batch, ensuring all tensors have consistent dimensions.

    Args:
        batch: A list of dictionaries, each containing:
            - input_ids: Token IDs for the input sequence
            - labels: Token IDs for the target sequence
            - attention_mask: Attention mask for the input sequence

    Returns:
        A dictionary containing batched and padded tensors for:
            - input_ids: Padded token IDs
            - labels: Padded token IDs with IGNORE_INDEX for padding tokens
            - attention_mask: Binary mask indicating non-padding tokens
    """
    # Extract each component from the batch
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad all sequences in the batch
    padded_input_ids = pad_sequence(input_ids, padding_value=0)  # 0 is typically pad_token_id
    padded_labels = pad_sequence(labels, padding_value=IGNORE_INDEX)
    padded_attention_mask = pad_sequence(attention_masks, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_mask,
    }


def main() -> None:
    """Main function to train the Self-RAG Critic model.

    This function:
    1. Loads the configuration from a YAML file
    2. Sets up logging
    3. Initializes the model and tokenizer
    4. Prepares the training dataset
    5. Configures and runs the training process
    6. Saves the trained model and tokenizer
    """
    try:
        # Load configuration from YAML file
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as cf:
            cfg_dict = yaml.safe_load(cf)
        config = Config(**cfg_dict)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
        )

        logging.info("Initializing model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(config)

        logging.info("Loading training dataset...")
        dataset = CriticDataset(config.data_path, tokenizer, config)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            gradient_checkpointing=True,
            bf16=True,
            report_to=None,
            logging_dir="./logs",
            save_safetensors=True,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )

        # Start training
        logging.info("Starting training...")
        trainer.train()

        # Save the final model
        logging.info(f"Training completed. Saving model to {config.output_dir}")
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        logging.info("Training finished successfully!")

    except Exception as e:
        logging.error(f"An error occurred during training: {e!s}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
