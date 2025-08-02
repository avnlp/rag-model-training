"""Self-RAG Generator Training Script.

The code is based on the implementation from: https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/finetune.py

This script orchestrates the complete training pipeline for a Self-RAG Generator model.
It handles the entire workflow from data loading to model training and saving.

Configuration:
- Training parameters are specified in generator_train_config.yaml
- Model and tokenizer are loaded from Hugging Face Hub
- Training is performed using Accelerate for distributed training
- Checkpointing and logging are enabled for model checkpoints and training progress
"""

import copy
import json
import logging
import math
import os
import random
from functools import partial
from types import SimpleNamespace

import datasets
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    get_scheduler,
)

logger = get_logger(__name__)

PROMPT_DICT = {
    "prompt_input": ("### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"),
    "prompt_no_input": ("### Instruction:\n{instruction}\n\n### Response:\n"),
}


def load_config(config_path="generator_train_config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Set default values matching original argparse defaults
    defaults = {
        "dataset_name": None,
        "dataset_config_name": None,
        "train_file": None,
        "model_name_or_path": None,
        "config_name": None,
        "use_lora": False,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "max_seq_length": 512,
        "per_device_train_batch_size": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "num_train_epochs": 3,
        "max_train_steps": None,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0,
        "output_dir": None,
        "seed": None,
        "preprocessing_num_workers": None,
        "overwrite_cache": False,
        "checkpointing_steps": None,
        "logging_steps": None,
        "resume_from_checkpoint": None,
        "with_tracking": False,
        "report_to": "all",
        "low_cpu_mem_usage": False,
        "use_special_tokens": False,
    }

    # Create namespace object
    args = SimpleNamespace()

    # Apply defaults
    for key, value in defaults.items():
        setattr(args, key, value)

    # Override with YAML config
    for key, value in config_dict.items():
        setattr(args, key, value)

    return args


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_seq_length,
        truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    print(input_ids_lens)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, context_markups=None):
    """Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space

    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )
    source_text = (
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    )
    target_text = example["output"] + tokenizer.eos_token
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[: source_len - 1] = -100

    if context_markups is not None:
        context_start = False
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len + j] == context_markups[0]
                start_idx = j + source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx = start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx + 1 : end_idx] = -100
                context_start = False
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        msg = "messages field is empty."
        raise ValueError(msg)

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                msg = "Invalid role: {}".format(message["role"])
                raise ValueError(msg)
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main():
    # Load configuration from YAML
    args = load_config()

    # Convert scheduler type string to enum
    args.lr_scheduler_type = SchedulerType(args.lr_scheduler_type)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        msg = "Need either a dataset name or a training file."
        raise ValueError(msg)
    elif args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in [
            "json",
            "jsonl",
        ], "`train_file` should be a json/jsonl file."

    # Initialize the accelerator
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer | LlamaTokenizerFast):
        # Collect special tokens for Llama-3 style models
        special_token_dict = {}
        if args.use_special_tokens:
            special_token_dict["additional_special_tokens"] = [
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
        # Llama 3 does not ship with a pad token - add one if necessary
        if tokenizer.pad_token is None:
            special_token_dict["pad_token"] = "<pad>"

        num_added_tokens = tokenizer.add_special_tokens(special_token_dict) if special_token_dict else 0

        # Track paragraph markup IDs for later masking
        context_markups = [
            tokenizer.convert_tokens_to_ids("<paragraph>"),
            tokenizer.convert_tokens_to_ids("</paragraph>"),
        ]

        if not args.use_special_tokens:
            assert num_added_tokens in [
                0,
                1,
            ], "Only the pad token should have been added."
        else:
            assert num_added_tokens > 10, "Expected additional special tokens to be added."
    else:
        msg = "Only Llama-based checkpoints are supported by this script."
        raise ValueError(msg)
    # Non-Llama tokenizers are not supported beyond this point

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        logger.info("Initializing LORA model...")
        modules_to_save = ["embed_tokens"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            modules_to_save=modules_to_save,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        context_markups=context_markups if args.use_special_tokens is True else None,
    )

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())

    train_dataset = lm_datasets["train"]

    with open("processed.json", "w") as outfile:
        new_data = []
        for item in train_dataset:
            labels = [int(i) for i in item["labels"]]
            input_ids = [int(i) for i in item["input_ids"]]
            new_data.append({"labels": labels, "input_ids": input_ids})
        json.dump(new_data, outfile)
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)

        state_dict = accelerator.get_state_dict(model)
        if args.use_lora:
            # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
            # and has its own save_pretrained function for only saving lora modules.
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        else:
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )


if __name__ == "__main__":
    main()
