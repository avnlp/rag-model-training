"""
Critic Data Processor for Self-RAG

This module processes and combines different types of critic tokens (groundness, relevance, retrieval, utility) into a unified training dataset for the Critic model.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/critic/gpt4_reward/combine_chat_gpt_reward.py

The implementation has been modified to work with the Earnings Call Dataset.
"""

import json
import random
from collections import Counter

import yaml

PROMPT_DICT = {
    "ground_instruction": (
        "You will be given a task instruction, evidence, and output. Your objective is to assess "
        "the extent to which the output is supported by the information presented in the evidence.\n"
        "Rate the level of support on a scale from 1 (Ignore/Contradictory), 2 (Little support), "
        "3 (Partially supported), 4 (Mostly supported), 5 (Fully supported)."
    ),
    "ground_input": (
        "##\nTask instruction: {instruction}\nEvidence: {evidence}\nOutput: {output}"
    ),
    "retrieval_instruction": (
        "When provided with instruction, please evaluate whether seeking additional information "
        "from external sources such as the web (e.g., Wikipedia) aids in producing a more "
        "comprehensive response. Respond with either [Retrieval] or [No Retrieval]."
    ),
    "retrieval_input": "Task instruction: {instruction}",
    "relevance_instruction": (
        "When given instruction and evidence, evaluate whether the evidence is relevant to the "
        "instruction and provides valuable information for generating meaningful responses.\n"
        "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to "
        "indicate irrelevance."
    ),
    "relevance_input": ("Task instruction: {instruction}\nEvidence: {evidence}"),
    "utility_instruction": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and "
        "informative answer to the query, from 1 (lowest) - 5 (highest). We call this score "
        "perceived utility.\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative response "
        "to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some "
        "minor improvements such as discussing more detailed information, having better structure "
        "of the response, or improving coherence.\n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are "
        "needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not "
        "relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
    ),
    "utility_input": ("Task instruction: {instruction}\nOutput: {output}"),
}


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_json_data(file_paths):
    """Load data from multiple JSON files"""
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.extend(json.load(f))
    return data


def create_utility_data(input_data):
    """Create utility training data from processed utility tokens"""
    print("Creating utility data...")
    processed_data = []

    for item in input_data:
        input_item = item["input"]
        raw_output = item["raw_output"]

        # Handle empty scores
        if item["score"] == "":
            item["score"] = raw_output.split("\n")[0]

        output = item["score"]
        if output not in [1, 2, 3, 4, 5] or len(str(output)) == 0:
            continue

        label = "[Utility:{}]".format(output)
        processed_data.append(
            {
                "instruction": PROMPT_DICT["utility_instruction"],
                "input": PROMPT_DICT["utility_input"].format_map(input_item),
                "output": label,
                "task": "utility",
            }
        )

    print(f"Utility data sample: {processed_data[-1] if processed_data else 'None'}")
    print(f"Total utility data: {len(processed_data)}")
    print(
        f"Utility distribution: {Counter([item['output'] for item in processed_data])}"
    )
    return processed_data


def create_retrieval_data_input_only(input_data):
    """Create retrieval data from input-only format"""
    print("Creating input-only retrieval data...")
    processed_data = []

    for item in input_data:
        try:
            # Extract instruction from input format
            instruction = item["input"].split("##\nTask instruction: ")[1]
            input_item = {"instruction": instruction}
            output = item["output"]

            if len(str(output)) == 0:
                continue

            if "Yes" in output:
                output = "[Retrieval]"
            elif "No" in output:
                output = "[No Retrieval]"
            else:
                continue

            processed_data.append(
                {
                    "instruction": PROMPT_DICT["retrieval_instruction"],
                    "input": PROMPT_DICT["retrieval_input"].format_map(input_item),
                    "output": output,
                    "task": "retrieval",
                }
            )
        except (IndexError, KeyError):
            continue

    print(
        f"Input-only retrieval data sample: {processed_data[-1] if processed_data else 'None'}"
    )
    print(f"Total input-only retrieval data: {len(processed_data)}")
    print(
        f"Input-only retrieval distribution: {Counter([item['output'] for item in processed_data])}"
    )
    return processed_data


def create_groundness_data(input_data):
    """Create groundness training data from processed groundness tokens"""
    print("Creating groundness data...")
    processed_data = []

    for item in input_data:
        input_item = item["input"]
        raw_output = item["raw_output"]

        # Handle empty scores
        if item["score"] == "":
            item["score"] = raw_output.split("\n")[0]

        # Clean up score
        if len(item["score"]) > 0 and item["score"][-1] == " ":
            item["score"] = item["score"][:-1]

        if len(item["score"]) == 0 or item["score"] not in [
            "[No support / Contradictory]",
            "[Fully supported]",
            "[Partially supported]",
        ]:
            continue

        processed_data.append(
            {
                "instruction": PROMPT_DICT["ground_instruction"],
                "input": PROMPT_DICT["ground_input"].format_map(input_item),
                "output": item["score"],
                "task": "groundness",
            }
        )

    print(f"Groundness data sample: {processed_data[-1] if processed_data else 'None'}")
    print(f"Total groundness data: {len(processed_data)}")
    print(
        f"Groundness distribution: {Counter([item['output'] for item in processed_data])}"
    )
    return processed_data


def create_relevance_data(input_data, relevant_sampling_rate=0.7):
    """Create relevance training data from processed relevance tokens"""
    print("Creating relevance data...")
    processed_data = []

    for item in input_data:
        input_item = item["input"]
        raw_output = item["raw_output"]

        # Handle empty scores
        if item.get("rating", "") == "":
            # Try to extract from raw output
            if "[Relevant]" in raw_output:
                item["rating"] = "[Relevant]"
            elif "[Irrelevant]" in raw_output:
                item["rating"] = "[Irrelevant]"
            else:
                continue

        # Clean up score
        score = item["rating"]
        if len(score) > 0 and score[-1] == " ":
            score = score[:-1]

        if score not in ["[Relevant]", "[Irrelevant]"]:
            continue

        # Apply sampling to reduce [Relevant] examples
        if score == "[Relevant]" and random.random() > relevant_sampling_rate:
            continue

        processed_data.append(
            {
                "instruction": PROMPT_DICT["relevance_instruction"],
                "input": PROMPT_DICT["relevance_input"].format_map(input_item),
                "output": score,
                "task": "relevance",
            }
        )

    print(f"Relevance data sample: {processed_data[-1] if processed_data else 'None'}")
    print(f"Total relevance data: {len(processed_data)}")
    print(
        f"Relevance distribution: {Counter([item['output'] for item in processed_data])}"
    )
    return processed_data


def main():
    # Load configuration
    config = load_config("create_final_dataset_config.yaml")

    # Set random seed for reproducibility
    if config.get("random_seed"):
        random.seed(config["random_seed"])

    # Load data from different token files
    print("Loading data files...")

    # Load utility data
    utility_data = []
    if config.get("utility_files"):
        utility_data = load_json_data(config["utility_files"])

    # Load retrieval data
    retrieval_data = []
    if config.get("retrieval_files"):
        retrieval_data = load_json_data(config["retrieval_files"])

    # Load groundness data
    groundness_data = []
    if config.get("groundness_files"):
        groundness_data = load_json_data(config["groundness_files"])

    # Load relevance data
    relevance_data = []
    if config.get("relevance_files"):
        relevance_data = load_json_data(config["relevance_files"])

    # Create training data
    print("\nProcessing data...")
    final_train_data = []

    if utility_data:
        final_train_data += create_utility_data(utility_data)

    if retrieval_data:
        final_train_data += create_retrieval_data_input_only(retrieval_data)

    if relevance_data:
        final_train_data += create_relevance_data(
            relevance_data, config.get("relevant_sampling_rate", 0.7)
        )

    if groundness_data:
        final_train_data += create_groundness_data(groundness_data)

    # Filter out problematic entries
    final_train_data = [
        item
        for item in final_train_data
        if "### Response:" not in item["input"]
        and "### Response:" not in item["instruction"]
    ]

    # Shuffle data
    random.shuffle(final_train_data)

    # Split into train and dev
    dev_size = config.get("dev_size", 1500)
    train_data = final_train_data[dev_size:]
    dev_data = final_train_data[:dev_size]

    # Save final datasets
    output_prefix = config["output_file_prefix"]

    with open(f"{output_prefix}_train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(f"{output_prefix}_dev.json", "w") as f:
        json.dump(dev_data, f, indent=2)

    print("\nFinal dataset created:")
    print(f"Total examples: {len(final_train_data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Development examples: {len(dev_data)}")
    print(f"Task distribution: {Counter([item['task'] for item in final_train_data])}")
    print(f"Files saved: {output_prefix}_train.json, {output_prefix}_dev.json")


if __name__ == "__main__":
    main()
