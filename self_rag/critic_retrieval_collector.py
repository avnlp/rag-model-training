"""
Retrieval Critic Data Collector for Self-RAG

This module adds retrieval decision annotations to the Critic model training data.
It uses an LLM to determine whether a generated output requires additional retrieval.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/critic/gpt4_reward/chatgpt_need_retrieval.py

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
    "You will be provided with an instruction, evidence, and output sentence. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn't require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
    "##\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
    "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.\n"
    "Evidence: Word embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.\n"
    "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
    "Rating: [Retrieval]\n"
    "Explanation: The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how it works. Therefore, we need to retrieve other evidence to verify whether the output is actually correct or not.\n"
    "###\nInstruction: {instruction}\n"
    "Evidence: {evidence}\n"
    "Output: {target_output}\n"
    "Rating: "
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        return [obj for obj in jsonl_f]


def postprocess(raw_output):
    """Extract decision token and explanation from raw output"""
    # First try to extract decision token
    decision_token = None
    for token in [
        "[Yes]",
        "[No]",
        "[No Retrieval]",
        "[Retrieval]",
        "[Continue to Use Evidence]",
    ]:
        if token in raw_output:
            decision_token = token
            break

    # Then extract explanation
    explanation = ""
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:", 1)[1].strip()
    elif "Explanation:" in raw_output:
        explanation = raw_output.split("Explanation:", 1)[1].strip()

    return decision_token, explanation


def process_input(example):
    """Select and format the appropriate prompt template"""
    return PROMPT.format(
        instruction=example.get("instruction", ""),
        evidence=example.get("evidence", ""),
        target_output=example.get("target_output", example.get("output", "")),
    )


def main():
    config = load_config("adding_retrieval_tokens_config.yaml")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=config["api_key"],
        organization=config.get("org_name"),
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )

    # Load input files
    examples = load_jsonlines(config["input_file"])

    # Process ID fields
    for item in examples:
        if "id" not in item and "q_id" in item:
            item["id"] = item["q_id"]

    # Process examples
    result_list = []
    output_file = config["output_file_name"]

    for idx, example in enumerate(tqdm(examples, desc="Processing")):
        try:
            # Generate prompt
            prompt = process_input(example)

            # Call API
            response = completions_with_backoff(
                client,
                model=config.get("model_name", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            # Process response
            raw_output = response.choices[0].message.content
            decision_token, explanation = postprocess(raw_output)

            # Store result
            result = {
                "input": example,
                "decision_token": decision_token,
                "explanation": explanation,
                "raw_output": raw_output,
            }
            result_list.append(result)

            # Periodic logging
            if idx % 20 == 0:
                print(f"\n--- Example {idx} ---")
                print(f"Instruction: {example.get('instruction', '')[:100]}...")
                print(f"Decision: {decision_token}")
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
