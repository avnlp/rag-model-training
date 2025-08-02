"""Utility Critic Data Collector for Self-RAG.

This module collects utility (IsUse) annotations for the Critic model training data.
It uses an LLM to evaluate how helpful and informative the generated output is for the given instruction.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/critic/gpt4_reward/chatgpt_utility.py

The implementation has been modified to work with the Earnings Call Dataset.

Changes from the original implementation:
1. Rewritten to process the Earnings Call Dataset format.
This includes updating the instruction format and evidence prompts.
2. Updated to work with the latest OpenAI API spec (>= 1.0.0).
"""

import json

import backoff
import jsonlines
import openai
import yaml
from openai import OpenAI
from tqdm import tqdm


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


PROMPT = (
    "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n\n"
    "The detailed criterion is as follows:\n"
    "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
    "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
    "3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
    "2: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
    "1: The response is barely on-topic or completely irrelevant.\n"
    "##\n Instruction: Who is the current prime minister of the UK as of 2023?\n"
    "Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.\n"
    "Perceived utility: 2\n"
    "Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.\n\n"
    "##\nInstruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.\n"
    "Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.\n"
    "Perceived utility: 3\n"
    "Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. \n\n"
    "##\nInstruction: {instruction}\n"
    "Output:{output}\n"
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        return list(jsonl_f)


def preprocess(item):
    return PROMPT.format(instruction=item["instruction"], output=item["output"])


def postprocess(raw_output):
    if "\nExplanation:" in raw_output:
        parts = raw_output.split("\nExplanation:", 1)
        score_string = parts[0]
        explanation = parts[1].strip()
        for i in range(1, 6):
            if str(i) in score_string:
                return i, explanation
    return None, ""


def main():
    config = load_config("adding_utility_tokens_config.yaml")

    client = OpenAI(
        api_key=config["api_key"],
        organization=config["org_name"],
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )

    examples = load_jsonlines(config["input_file"])

    for item in examples:
        if item.get("input"):
            item["instruction"] = f"{item['instruction']} {item['input']}"

    result_list = []
    output_file = config["output_file_name"]

    for idx, example in enumerate(tqdm(examples)):
        try:
            response = completions_with_backoff(
                client,
                model=config["model_name"],
                messages=[{"role": "user", "content": preprocess(example)}],
                max_tokens=200,
            )

            content = response.choices[0].message.content
            score, explanation = postprocess(content)

            result = {
                "input": example,
                "score": score,
                "explanation": explanation,
                "raw_output": content,
            }
            result_list.append(result)

            if idx % 20 == 0:
                print(f"\nInput: {example['instruction']}")
                print(f"Output: {example['output']}")
                print(f"Score: {score} ({explanation})")

            if idx % 100 == 0:
                with open(f"{output_file}.tmp", "w") as f:
                    json.dump(result_list, f)

        except openai.APIError as e:
            print(f"API Error on example {idx}: {e}")
            result_list.append({"input": example, "error": str(e)})

    with open(output_file, "w") as f:
        json.dump(result_list, f)


if __name__ == "__main__":
    main()
