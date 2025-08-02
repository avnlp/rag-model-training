# This code is based on the implementation provided at: https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/run_classifier.py

import logging
import math
import os

import datasets
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

logger = logging.getLogger(__name__)


def preprocess_dataset(args, raw_datasets):
    column_names = raw_datasets[args.train_column].column_names

    question_column = args.question_column
    if question_column not in column_names:
        msg = f"--question_column' value '{args.question_column}' needs to be one of: {', '.join(column_names)}"
        raise ValueError(msg)

    answer_column = args.answer_column
    if answer_column not in column_names:
        msg = f"--answer_column' value '{args.answer_column}' needs to be one of: {', '.join(column_names)}"
        raise ValueError(msg)

    return question_column, answer_column


def preprocess_features_function(examples, args, raw_datasets, tokenizer):
    question_column, answer_column = preprocess_dataset(args, raw_datasets)

    max_answer_length = args.max_answer_length
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    examples[question_column] = [f"{q.strip()}" for q in examples[question_column]]

    model_inputs = tokenizer(
        examples[question_column],
        truncation=True,
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    targets = examples[answer_column]

    labels = tokenizer(
        text_target=targets,
        max_length=max_answer_length,
        padding=padding,
        truncation=True,
    )

    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    model_inputs["example_id"] = []
    labels_out = []

    for i in range(len(model_inputs["input_ids"])):
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_out.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = labels_out
    return model_inputs


def prepare_scheduler(args, accelerator, dataloader, optimizer, max_train_steps, train_epoch):
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if max_train_steps is None:
        max_train_steps = train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if overrode_max_train_steps:
        max_train_steps = train_epoch * num_update_steps_per_epoch
    train_epoch = math.ceil(max_train_steps / num_update_steps_per_epoch)

    return max_train_steps, train_epoch, lr_scheduler


option_to_label = {
    "A": 0,
    "B": 1,
    "C": 2,
}

label_to_option = {
    0: "A",
    1: "B",
    2: "C",
}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert scheduler type string to enum
    if "lr_scheduler_type" in config:
        config["lr_scheduler_type"] = SchedulerType(config["lr_scheduler_type"])

    return config


def main():
    config = load_config("train_adaptive_rag_config.yaml")
    # Convert dict to object
    args = type("Args", (object,), config)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "logs.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    data_files["train"] = args.train_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = raw_datasets[args.train_column]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_features_function,
            fn_kwargs={
                "args": args,
                "raw_datasets": raw_datasets,
                "tokenizer": tokenizer,
            },
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
    train_dataloader = DataLoader(
        train_dataset_for_model,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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

    model, optimizer = accelerator.prepare(model, optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train
    args.max_train_steps, args.num_train_epochs, lr_scheduler_train = prepare_scheduler(
        args,
        accelerator,
        train_dataloader,
        optimizer,
        args.max_train_steps,
        args.num_train_epochs,
    )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler_train.step()
                optimizer.zero_grad()

                logger.info(f"Loss:{loss} ")
                total_loss = total_loss + loss.cpu().detach().float()

            logger.info(tokenizer.batch_decode(batch["input_ids"][:1], skip_special_tokens=True))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        logger.info(f"Epoch %d Loss:{total_loss / len(train_dataloader)} ", epoch)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
