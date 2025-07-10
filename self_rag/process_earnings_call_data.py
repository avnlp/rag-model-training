"""
Process the Earnings Call Dataset to the specified format.

This file processes the Earnings Call Dataset into the specified format and saves it to a JSONLines file.

The data is formatted as follows:
{
    "id": str # unique instance id,
    "instruction": str, # input instruction
    "input": str # input question
    "evidence": str, # context for the answer
    "output": str, # answer
}
"""

import random
from collections import OrderedDict

import jsonlines
import yaml
from datasets import load_dataset

# Load configuration from YAML file
with open("process_earnings_call_data_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Load dataset from Hugging Face
dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])

# Process and format dataset
formatted_data = []
for idx, item in enumerate(dataset):
    formatted_item = OrderedDict(
        [
            ("id", f"{config['data_prefix']}_{idx}"),
            ("instruction", config["task_instruction"]),
            ("input", item["question"]),
            ("evidence", item["transcript"]),
            ("output", item["answer"]),
        ]
    )
    formatted_data.append(formatted_item)

# Apply random sampling if specified in config
if config.get("n_samples"):
    formatted_data = random.sample(formatted_data, k=config["n_samples"])

# Save to JSONLines file
with jsonlines.open(config["output_file"], "w") as writer:
    writer.write_all(formatted_data)

print(
    f"Successfully processed {len(formatted_data)} records to {config['output_file']}"
)
