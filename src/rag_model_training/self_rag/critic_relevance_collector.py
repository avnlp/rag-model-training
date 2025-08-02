"""Relevance Critic Data Collector for Self-RAG.

This module adds relevance (IsRel) annotations to the Critic model training data.
It uses an LLM to evaluate how relevant the provided evidence is to the given input.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/critic/gpt4_reward/chatgpt_relevance.py

The implementation has been modified to work with the Earnings Call Dataset.

Changes from the original implementation:
1. Rewritten to process the Earnings Call Dataset format.
This includes updating the instruction format and evidence prompts.
2. Updated to work with the latest OpenAI API spec (>= 1.0.0).
"""

import json
import random

import backoff
import jsonlines
import openai
import yaml
from openai import OpenAI
from tqdm import tqdm


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


KNOWLEDGE_INSTRUCTION = (
    "Please answer the following questions using the shortest most factually accurate response."
    " For example, if the question asks 'What is the capital of France?', you can simply reply with 'Paris'."
)


PROMPT = (
    "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
    "When there are preceding sentences, your focus should be on the sentence that comes after them. "
    "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
    "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
    "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
    "Input: Earth rotating causes\n"
    "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
    "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
    "Rating: [Relevant]\n"
    "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
    "###\nInstruction: age to run for us house of representatives\n\n"
    "Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.\n\n"
    "Rating: [Irrelevant]\n"
    "Explanation: The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.\n\n"
    "###\nInstruction: {instruction}\n\n"
    "Evidence: {evidence}\n\n"
    "Rating:"
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        return list(jsonl_f)


def postprocess(raw_output):
    # Check if output contains rating tags
    if "[Relevant]" in raw_output:
        rating = "Relevant"
    elif "[Irrelevant]" in raw_output:
        rating = "Irrelevant"
    else:
        rating = "Unknown"

    # Extract explanation
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:", 1)[1].strip()
    else:
        explanation = "No explanation provided"

    return rating, explanation


def process_input(example):
    return PROMPT.format(instruction=example["instruction"], evidence=example["evidence"])


def main():
    config = load_config("adding_relevance_tokens_config.yaml")

    client = OpenAI(
        api_key=config["api_key"],
        organization=config["org_name"],
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )

    # Load input file
    examples = load_jsonlines(config["input_file"])

    # Apply sampling if requested
    n_samples = config.get("n")
    if n_samples and len(examples) > n_samples:
        examples = random.sample(examples, n_samples)

    # Preprocess examples
    for example in examples:
        # Handle different field names
        if "output" not in example and "answers" in example:
            example["output"] = example["answers"][0] if isinstance(example["answers"], list) else example["answers"]

        if "target_output" not in example and "output" in example:
            example["target_output"] = example["output"]

        if "instruction" not in example and "question" in example:
            example["instruction"] = KNOWLEDGE_INSTRUCTION + " " + example["question"]

    # Filter out problematic responses
    examples = [e for e in examples if "As a language model, I cannot" not in e.get("output", "")]

    # Process examples
    result_list = []
    output_file = config["output_file_name"]

    for idx, example in enumerate(tqdm(examples)):
        try:
            # Generate prompt
            prompt = process_input(example)

            # Call API
            response = completions_with_backoff(
                client,
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            # Process response
            raw_output = response.choices[0].message.content
            rating, explanation = postprocess(raw_output)

            # Store result
            result = {
                "input": example,
                "rating": rating,
                "explanation": explanation,
                "raw_output": raw_output,
            }
            result_list.append(result)

            # Periodic logging
            if idx % 20 == 0:
                print(f"\n--- Example {idx} ---")
                print(f"Instruction: {example.get('instruction', '')}")
                print(f"Evidence: {example.get('evidence', '')[:200]}...")
                print(f"Rating: {rating}")
                print(f"Explanation: {explanation[:200]}...")

            # Periodic saving
            if idx % 100 == 0:
                with open(f"{output_file}.tmp", "w") as f:
                    json.dump(result_list, f)

        except openai.APIError as e:
            print(f"API Error on example {idx}: {e}")
            result_list.append({"input": example, "error": str(e)})

    # Save final results
    with open(output_file, "w") as f:
        json.dump(result_list, f)
    print(f"Processing complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
