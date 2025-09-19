# This code is based on the implementation from: https://github.com/dCaples/AutoDidact/blob/main/autodidact.ipynb.

from pathlib import Path

import yaml
from unsloth import FastLanguageModel
from vllm import SamplingParams

from .evaluation import run_eval
from .rl_helpers import (
    build_reward_correctness_fn,
    get_qa_dataset,
    run_agent,
)

# Load configuration from YAML file
config_path = Path(__file__).parent / "inference_config.yaml"
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Model configuration
model_config = config["model"]
lora_config = config["lora"]
sampling_config = config["sampling"]
generation_sampling_config = config["generation_sampling"]
agent_config = config["agent"]

# Initialize model with specified parameters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["load_in_4bit"],
    fast_inference=model_config["fast_inference"],
    max_lora_rank=lora_config["rank"],
    gpu_memory_utilization=model_config["gpu_memory_utilization"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config["rank"],
    target_modules=lora_config["target_modules"],
    lora_alpha=lora_config["alpha"],
    use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
    random_state=lora_config["random_state"],
)


# Load training and test datasets
train_dataset, test_dataset = get_qa_dataset()

# Configure sampling parameters for generation
verifier_sampling_params = SamplingParams(
    temperature=sampling_config["temperature"],
    top_p=sampling_config["top_p"],
    max_tokens=sampling_config["max_tokens"],
)


def agentic_generate(
    prompts: list[str],
    generate_fn,
    max_generations: int = agent_config["max_generations"],
):
    """Generate responses using an agentic approach.

    Args:
        prompts (list[str]): List of input prompts for generation.
        generate_fn: Function to use for generating responses.
        max_generations (int): Maximum number of generations to produce.

    Returns:
        Generated responses from the agent.
    """
    return run_agent(generate_fn, tokenizer, prompts, max_generations)


def verifier_generate_fn(inputs):
    """Generate responses using the model with fast inference.

    Args:
        inputs: Input prompts for generation.

    Returns:
        Model-generated responses.
    """
    return model.fast_generate(
        inputs,
        sampling_params=verifier_sampling_params,
    )


# Attach agentic generation method to model
model.agentic_generate = agentic_generate

# Build reward function for correctness evaluation
reward_correctness = build_reward_correctness_fn(
    verifier_generate_fn,
    tokenizer,
)

sampling_params = SamplingParams(
    temperature=generation_sampling_config["temperature"],
    top_p=generation_sampling_config["top_p"],
    max_tokens=generation_sampling_config["max_tokens"],
)


def eval_generate_fn(inputs):
    return model.fast_generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=model.load_lora(config["lora"]["name"]),
    )


run_eval(
    generate_fn=eval_generate_fn,
    verify_fn=reward_correctness,
    tokenizer=tokenizer,
)
