import os
from pathlib import Path

import yaml
from rl_helpers import (
    build_reward_correctness_fn,
    get_qa_dataset,
    reward_formatting,
    run_agent,
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth_grpo_trainer_agent import UnslothGRPOConfig, UnslothGRPOTrainer
from vllm import SamplingParams

# Load configuration from YAML file
config_path = Path(__file__).parent / "training_config.yaml"
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Model configuration
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]
sampling_config = config["sampling"]
agent_config = config["agent"]

# Initialize model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["load_in_4bit"],
    fast_inference=model_config["fast_inference"],
    max_lora_rank=lora_config["rank"],
    gpu_memory_utilization=model_config["gpu_memory_utilization"],
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config["rank"],
    target_modules=lora_config["target_modules"],
    lora_alpha=lora_config["alpha"],
    use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
    random_state=lora_config["random_state"],
)

# Set WANDB project
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

# Load dataset
train_dataset, test_dataset = get_qa_dataset()

# Configure training arguments with values from YAML
training_args = UnslothGRPOConfig(
    use_vllm=training_config["use_vllm"],
    use_agentic_generate=training_config["use_agentic_generate"],
    learning_rate=training_config["learning_rate"],
    adam_beta1=training_config["adam_beta1"],
    adam_beta2=training_config["adam_beta2"],
    weight_decay=training_config["weight_decay"],
    warmup_ratio=training_config["warmup_ratio"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    optim=training_config["optim"],
    logging_steps=training_config["logging_steps"],
    bf16=is_bfloat16_supported() if training_config["bf16"] else False,
    fp16=not is_bfloat16_supported() if training_config["fp16"] else False,
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    num_generations=training_config["num_generations"],
    max_prompt_length=training_config["max_prompt_length"],
    max_completion_length=training_config["max_completion_length"],
    max_steps=training_config["max_steps"],
    save_steps=training_config["save_steps"],
    max_grad_norm=training_config["max_grad_norm"],
    report_to=training_config["report_to"],
    output_dir=training_config["output_dir"],
)

# Configure sampling parameters
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
    return run_agent(generate_fn, tokenizer, prompts, max_generations)


def verifier_generate_fn(inputs):
    return model.fast_generate(
        inputs,
        sampling_params=verifier_sampling_params,
    )


model.agentic_generate = agentic_generate

reward_correctness = build_reward_correctness_fn(
    verifier_generate_fn,
    tokenizer,
)

trainer = UnslothGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_correctness,
        reward_formatting,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
