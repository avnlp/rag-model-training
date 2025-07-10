"""
Critic Evaluator for Self-RAG

This module provides a class for running critic model evaluations using vLLM.

This code is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/generator/run_reward_vllm.py

The implementation has been rewritten to be modular and more efficient.
"""

import json
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT = {
    "ground_instruction": (
        "You will be given an task instruction, evidence, and output. Your objective is to assess the extent to which the output is supported by the information presented in the evidence.\n"
        "Rate the level of support on a scale from 1 (Ignore / Contradictory), 2 (Little support), 3 (Partially supported), 4 (Mostly supported), 5 (Fully supported)."
    ),
    "ground_input": "##\nTask instruction: {instruction}\nEvidence: {evidence}\nOutput: {output}",
    "ground_multi_instruction": (
        "You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n"
        "Use the following entailment scale to generate a score:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
    ),
    "ground_multi_input": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Output: {target_output}\n"
        "Evidence: {evidence}"
    ),
    "ground_multi_input_wo_preceding": (
        "Task instruction: {instruction}\nOutput: {target_output}\nEvidence: {evidence}"
    ),
    "retrieval_instruction": (
        "When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."
    ),
    "retrieval_input": "Task instruction: {instruction}",
    "retrieval_multi_instruction": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn't require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
    ),
    "retrieval_multi_input": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}"
    ),
    "multi_retrieval_three_way_instruction": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with  [No Retrieval]. \n"
        "If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
    ),
    "relevance_instruction": (
        "When given instruction and evidence, evaluate whether the evidence is relevant to the instruction and provides valuable information for generating meaningful responses.\n"
        "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
    ),
    "relevance_input": "Task instruction: {instruction}\nEvidence: {evidence}",
    "utility_instruction": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
    ),
    "utility_input": "Task instruction: {instruction}\nOutput: {output}",
}


class CriticEvaluator:
    """A class to handle critic model evaluations using vLLM."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 15,
        batch_size: int = 10,
        split: str = "train",
    ):
        """
        Initialize the CriticEvaluator.

        Args:
            model_name: Name or path of the critic model
            download_dir: Directory to download and cache the model
            device: Device to run the model on ('cuda' or 'cpu')
            max_new_tokens: Maximum number of tokens to generate
            batch_size: Batch size for processing
            split: Data split ('train' or 'test')
        """
        self.model_name = model_name
        self.download_dir = download_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.split = split

        # Initialize the model
        self.model = LLM(model=model_name, download_dir=download_dir)

    @staticmethod
    def _posprocess_output(answer: str) -> str:
        """Clean up the model output."""
        answer = answer.replace("</s>", "")
        answer = answer.replace("<unk>", "")
        answer = answer.replace("[PAD]", "")
        return answer.strip()

    def _call_model(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Call the model with a batch of prompts.

        Args:
            prompts: List of input prompts

        Returns:
            Tuple of (postprocessed_preds, raw_preds)
        """
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=self.max_new_tokens
        )
        preds = self.model.generate(prompts, sampling_params)
        raw_preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        postprocessed_preds = [self._posprocess_output(pred) for pred in raw_preds]
        return postprocessed_preds, raw_preds

    def _process_data(
        self,
        input_data: Dict,
        inst_mode: str,
        input_mode: str,
        multi_retrieval: bool = False,
    ) -> Tuple[str, str]:
        """
        Process input data into a prompt and expected output.

        Args:
            input_data: Input data dictionary
            inst_mode: Instruction mode key from PROMPT_DICT
            input_mode: Input mode key from PROMPT_DICT
            multi_retrieval: Whether to use multi-retrieval mode

        Returns:
            Tuple of (prompt, output)
        """
        if self.split == "train":
            prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(input_data)
            output = str(input_data["output"])
            return prompt, output
        else:
            instruction = PROMPT_DICT[inst_mode]
            if multi_retrieval and (
                input_data.get("sent_idx") == 0
                or "preceding_sentences" not in input_data
                or not isinstance(input_data["preceding_sentences"], str)
                or not input_data["preceding_sentences"]
            ):
                input_text = PROMPT_DICT.get(
                    f"{input_mode}_no_preceding", PROMPT_DICT[input_mode]
                ).format_map(input_data)
            else:
                input_text = PROMPT_DICT[input_mode].format_map(input_data)

            prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(
                {"instruction": instruction, "input": input_text}
            )
            return prompt, "None"

    def evaluate(
        self,
        input_file: str,
        task: str,
        inst_mode: str,
        input_mode: str,
        result_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run critic evaluation on input data.

        Args:
            input_file: Path to input JSON or JSONL file
            task: Task name (e.g., 'retrieval', 'utility', 'relevance', 'groundness')
            inst_mode: Instruction mode key from PROMPT_DICT
            input_mode: Input mode key from PROMPT_DICT
            result_file: Optional path to save results

        Returns:
            List of results with predictions
        """
        # Load input data
        if input_file.endswith(".json"):
            with open(input_file, "r") as f:
                input_data = json.load(f)
        else:
            input_data = []
            with open(input_file, "r") as f:
                for line in f:
                    input_data.append(json.loads(line))

        # Filter by task if in train mode
        if self.split == "train":
            input_data = [item for item in input_data if item.get("task") == task]

        results = []
        multi_retrieval = task in ["multi_retrieval", "groundness", "relevance"]

        # Process in batches
        for i in range(0, len(input_data), self.batch_size):
            batch = input_data[i : i + self.batch_size]

            # Process batch
            processed_batch = []
            for item in batch:
                prompt, output = self._process_data(
                    item, inst_mode, input_mode, multi_retrieval
                )
                processed_batch.append((prompt, output, item))

            # Get model predictions
            prompts = [p[0] for p in processed_batch]
            preds, raw_preds = self._call_model(prompts)

            # Process results
            for (_, output, item), pred, raw_pred in zip(
                processed_batch, preds, raw_preds
            ):
                result = item.copy()
                result["pred"] = pred

                if self.split == "train":
                    result["output"] = output
                    if pred and (pred == output or pred in output or output in pred):
                        result["correct"] = 1.0
                    else:
                        result["correct"] = 0.0

                results.append(result)

        # Save results if output file is provided
        if result_file:
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

        return results
