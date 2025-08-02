# The code is based on the implementation provided at: https://github.com/HuskyInSalt/CRAG/blob/main/scripts/train_evaluator.py

import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import T5ForSequenceClassification, T5Tokenizer, get_scheduler


def get_data(file):
    content = []
    label = []
    with open(file, encoding="utf-8") as f:
        for i in f.readlines()[:]:
            c, l = i.split("\t")
            content.append(c)
            label.append((int(l.strip()) - 0.5) * 2)
    return content, label


def data_preprocess(file, tokenizer, max_length):
    content, label = get_data(file)
    data = pd.DataFrame({"content": content, "label": label})
    train_data = tokenizer(
        data.content.to_list()[:],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    train_label = data.label.to_list()[:]
    return train_data, train_label


def main():
    with open("train_corrective_rag_config.yaml") as f:
        config = yaml.safe_load(f)

    # Set seeds
    SEED = config["training"]["seed"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["model_name"])
    model = T5ForSequenceClassification.from_pretrained(config["model"]["model_name"], num_labels=1)

    # Prepare data
    train_data, train_label = data_preprocess(
        config["data"]["train_file"],
        tokenizer,
        max_length=config["model"]["max_length"],
    )
    train_dataset = TensorDataset(train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
    train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    num_training_steps = config["training"]["num_epochs"] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=config["training"]["scheduler"]["name"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["scheduler"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    # Training setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gradient_clip = config["training"]["gradient_clip"]

    # Training loop
    for epoch in tqdm(range(config["training"]["num_epochs"])):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 10 == 0 and step != 0:
                print(f"step: {step}  loss: {total_loss / (step * config['training']['batch_size'])}")

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss.mean().backward()
            total_loss += loss.mean().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} avg_loss: {avg_train_loss}")

        save_path = f"{config['training']['save_path']}/ep{epoch}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
