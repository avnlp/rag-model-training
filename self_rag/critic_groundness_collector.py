"""
Groundness Critic Data Collector for Self-RAG

This module adds groundness (IsSup) annotations to the Critic model training data.
It uses an LLM to evaluate how well the generated output is supported by the provided evidence.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/critic/gpt4_reward/chatgpt_groundness.py

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


KNOWLEDGE_INSTRUCTION = (
    "Please answer the following questions using the shortest most factually accurate response."
    " For example, if the question asks 'What is the capital of France?', you can simply reply with 'Paris'."
)

PROMPT = (
    "You will receive an instruction, evidence, and output.\n"
    "Your task is to evaluate if the output is fully supported by the information provided in the evidence.\n"
    "Use the following entailment scale to generate a score:\n"
    "5: Fully supported - All information in output is supported by the evidence, or extractions from the evidence. This is a somewhat extreme case and is only applicable when the output and part of the evidence are almost identical.\n"
    "4: Mostly supported - Most of the information in the output is supported by the evidence, but there is some minor information that is not supported. In other words, if an output is a paraphrase of the evidence or a less concrete version of the descriptions of the evidence, it should be considered a 4.\n"
    "3: Partially supported - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a 3. If the output covers a lot of new information that is not discussed in the evidence, it should be 3.\n"
    "2: Little support - The output and evidence are only loosely related, and most of the information in the output isn't supported by the evidence.\n"
    "1: Ignore / Contradictory - The output completely ignores evidence or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n"
    "Make sure to not use any external information/knowledge to judge whether the output is true or not.\n"
    "Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
    "###\nInstruction: Explain the use of word embeddings in Natural Language Processing\n\n"
    "Output: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured. Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies. They allow for words to be processed as numerical values, giving machines an easier way to perform NLP tasks.\n\n"
    "Evidence: Word embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing and sentiment analysis.\n\n"
    "Score: 4\n"
    "Explanation: Although the wording isn't exactly same, the evidence verifies all of the claims in the output such as definitions and the use cases in NLP. Therefore, it should be rated as 4.\n\n"
    "###\nInstruction: {instruction}\n\n"
    "Output: {output}\n\n"
    "Evidence: {evidence}\n\n"
    "Score:"
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        return [obj for obj in jsonl_f]


def postprocess(raw_output):
    """Extract score and explanation from raw output"""
    # Try to find numeric score first
    score = None
    for i in range(1, 6):
        if f"Score: {i}" in raw_output or str(i) in raw_output:
            score = i
            break

    # If no numeric score, check for text ratings
    if score is None:
        if "[Fully supported]" in raw_output:
            score = 5
        elif "[Partially supported]" in raw_output:
            score = 3
        elif "[No support / Contradictory]" in raw_output:
            score = 1

    # Extract explanation
    explanation = ""
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:", 1)[1].strip()
    elif "Explanation:" in raw_output:
        explanation = raw_output.split("Explanation:", 1)[1].strip()

    return score, explanation


def process_input(example):
    """Select and format the appropriate prompt template"""
    return PROMPT.format(
        instruction=example.get("instruction", ""),
        output=example.get("output", ""),
        evidence=example.get("evidence", ""),
    )


def main():
    # Load configuration
    config = load_config("adding_groundness_config.yaml")

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

    # Preprocess examples
    for example in examples:
        # Handle different field names
        if "output" not in example and "answers" in example:
            example["output"] = (
                example["answers"][0]
                if isinstance(example["answers"], list)
                else example["answers"]
            )

        if "target_output" not in example and "output" in example:
            example["target_output"] = example["output"]

        if "instruction" not in example and "question" in example:
            data_type = example.get("q_id", "").split("_")[0]
            if data_type in KNOWLEDGE_INSTRUCTION:
                example["instruction"] = (
                    KNOWLEDGE_INSTRUCTION[data_type] + " " + example["question"]
                )
            else:
                example["instruction"] = example["question"]

    # Filter out problematic responses
    examples = [
        e
        for e in examples
        if "As a language model, I cannot" not in e.get("output", "")
    ]

    # Process examples
    result_list = []
    output_file = config["output_file_name"]

    for idx, example in enumerate(tqdm(examples, desc="Evaluating")):
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
            score, explanation = postprocess(raw_output)

            # Store result
            result = {
                "input": example,
                "score": score,
                "explanation": explanation,
                "raw_output": raw_output,
            }
            result_list.append(result)

            # Periodic logging
            if idx % 20 == 0:
                print(f"\n--- Example {idx} ---")
                print(f"Instruction: {example.get('instruction', '')[:100]}...")
                print(f"Evidence: {example.get('evidence', '')[:200]}...")
                print(f"Output: {example.get('output', '')[:200]}...")
                print(f"Score: {score}")
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
    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
