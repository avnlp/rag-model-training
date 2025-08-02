"""Data Preparation Pipeline for Self-RAG Generator Training.

The code in this file is based on the implementation from:
https://github.com/AkariAsai/self-rag/blob/main/data_creation/generator/create_retrieval_data.py and
https://github.com/AkariAsai/self-rag/blob/main/data_creation/generator/postprocess_data.py.

This code creates the training data for the Self-RAG generator model.
The following steps are performed:
1. Load the training data from a JSONL file.
2. Run the retriever for the different questions in the training data.
3. Evaluate the retrieved documents using the critic model.
4. Create the training data for the generator model with the filtered retrieved documents. This includes the special tokens for retrieval, relevance, groundness and utility.
5. Save the training data to a JSON file.
"""

import json
import logging
import os
from typing import Any

import jsonlines
import spacy
import yaml
from retrieval.retrieval_utils import create_retrieval_input, run_retrieval
from tqdm import tqdm

from .critic_evaluator import CriticEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]"]
ground_tokens_names = [
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]
utility_tokens_names = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]
other_special_tokens = ["<s>", "</s>", "[PAD]", "<unk>", "<paragraph>", "</paragraph>"]


def postprocess(pred: str) -> str:
    """Remove special tokens and extra whitespace."""
    special_tokens = (
        rel_tokens_names + retrieval_tokens_names + ground_tokens_names + utility_tokens_names + other_special_tokens
    )
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "").replace("<unk>", "")
    pred = pred.strip()
    pred = pred.replace("  ", " ")
    return pred.strip()


def save_file_jsonl(data: list[dict], fp: str) -> None:
    """Save a list of dicts to a JSONL file."""
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)


def _get_spacy_model():
    spacy_model = spacy.load("en_core_web_sm")
    return spacy_model


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences using spaCy."""
    doc = _get_spacy_model()(paragraph)
    return [sent.text for sent in doc.sents]


def convert_score_to_utility_token(pred: str) -> str | None:
    if not pred:
        return None
    for i in ["1", "2", "3", "4", "5"]:
        if i in pred:
            return f"[Utility:{i}]"
    if pred[0] != "[":
        pred = "[" + pred
    return pred if pred in ground_tokens_names else None


def convert_score_to_retrieval_token(pred: str) -> str | None:
    if not pred:
        return None
    if pred[0] != "[":
        pred = "[" + pred
    mapping = {
        "[Yes]": "[Retrieval]",
        "[No]": "[No Retrieval]",
        "Yes": "[Retrieval]",
        "No": "[No Retrieval]",
    }
    if pred in mapping:
        return mapping[pred]
    if pred in ["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]:
        return pred
    return "[No Retrieval]"


def convert_score_to_groundness(pred: str) -> str | None:
    if not pred:
        return None
    if pred[0] != "[":
        pred = "[" + pred
    if pred in [
        "[No support / Contradictory]",
        "[Partially supported]",
        "[Fully supported]",
    ]:
        return pred
    if pred in ["4", "5"]:
        return "[Fully supported]"
    return None


def postprocess_relevance_reward_token(pred: str) -> str | None:
    if not pred:
        return None
    if "Relevant" in pred:
        return "[Relevant]"
    if "Irrelevant" in pred:
        return "[Irrelevant]"
    return None


class GeneratorDataPreparation:
    """Main class to handle the generator data preparation pipeline."""

    def __init__(self, config_path: str):
        """Initialize the data preparation pipeline.

        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.config["output_dir"] = os.path.expandvars(self.config["output_dir"])
        os.makedirs(self.config["output_dir"], exist_ok=True)

        # Process all string values for environment variables
        for key, value in self.config.items():
            if isinstance(value, str):
                self.config[key] = os.path.expandvars(value)

        # Process nested paths in intermediate_files
        if "intermediate_files" in self.config:
            for key, path in self.config["intermediate_files"].items():
                # Replace ${output_dir} with the actual output_dir
                path = path.replace("${output_dir}", self.config["output_dir"])
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.config["intermediate_files"][key] = path

    def run_critic_evaluation(
        self,
        input_file: str,
        task: str,
        output_file: str | None,
        inst_mode: str,
        input_mode: str,
    ) -> list[dict]:
        """Run critic model evaluation for a specific task.

        Args:
            input_file: Path to input JSON or JSONL file
            task: Task name (e.g., 'retrieval', 'utility', 'relevance', 'groundness')
            output_file: Optional path to save results
            inst_mode: Instruction mode key from PROMPT_DICT
            input_mode: Input mode key from PROMPT_DICT

        Returns:
            List of results with predictions
        """
        # Initialize the critic evaluator
        evaluator = CriticEvaluator(
            model_name=self.config["critic_model"],
            max_new_tokens=15,
            batch_size=self.config.get("batch_size", 10),
        )

        # Run evaluation
        results = evaluator.evaluate(
            input_file=input_file,
            task=task,
            inst_mode=inst_mode,
            input_mode=input_mode,
            result_file=output_file,
        )

        return results

    def _output_path(self, key: str) -> str:
        """Shortcut to intermediate file path from config."""
        return self.config["intermediate_files"][key]

    def _run_and_store(
        self,
        task: str,
        out_key: str,
        inst_mode: str,
        input_mode: str,
        input_file: str | None = None,
    ) -> str:
        """Run critic evaluation for a simple single-file task and return output path."""
        if input_file is None:
            input_file = self.config["input_file"]
        output_path = self._output_path(out_key)
        self.run_critic_evaluation(
            input_file=input_file,
            task=task,
            output_file=output_path,
            inst_mode=inst_mode,
            input_mode=input_mode,
        )
        return output_path

    def _run_batch_evaluation(
        self,
        task: str,
        inst_mode: str,
        input_mode: str,
        prompt_data_dir: str,
        output_dir_key: str,
        output_prefix: str,
        desc: str,
    ) -> list[str]:
        """Run critic evaluation for each batch file in a directory."""
        output_dir = self._output_path(output_dir_key)
        os.makedirs(output_dir, exist_ok=True)
        batch_files = [f for f in os.listdir(prompt_data_dir) if f.startswith("prompt_data_batch_")]
        outputs: list[str] = []
        for batch_file in tqdm(batch_files, desc=desc):
            batch_num = batch_file.split("_")[-1].split(".")[0]
            out_file = os.path.join(output_dir, f"{output_prefix}_{batch_num}.json")
            self.run_critic_evaluation(
                input_file=os.path.join(prompt_data_dir, batch_file),
                task=task,
                output_file=out_file,
                inst_mode=inst_mode,
                input_mode=input_mode,
            )
            outputs.append(out_file)
        return outputs

    def prepare_initial_retrieval_data(self) -> str:
        """Prepare data for initial retrieval and run critic evaluation."""
        logging.info("Step 1: Running initial retrieval critic evaluation")
        return self._run_and_store(
            task="retrieval",
            out_key="initial_retrieval_preds",
            inst_mode="retrieval_instruction",
            input_mode="retrieval_input",
        )

    def prepare_multi_retrieval_data(self) -> str:
        """Prepare data for multi-step retrieval and run critic evaluation."""
        logging.info("Step 2: Running multi-retrieval critic evaluation")
        return self._run_and_store(
            task="multi_retrieval",
            out_key="multi_retrieval_preds",
            inst_mode="retrieval_multi_instruction",
            input_mode="retrieval_multi_input",
        )

    def prepare_utility_data(self) -> str:
        """Wrapper for utility critic evaluation."""
        logging.info("Step 3: Running utility critic evaluation")
        return self._run_and_store(
            task="utility",
            out_key="utility_preds",
            inst_mode="utility_instruction",
            input_mode="utility_input",
        )

    def run_retrieval_pipeline(self, initial_retrieval_output: str, multi_retrieval_output: str) -> dict:
        """Run the complete retrieval pipeline."""
        print("Step 4: Running retrieval pipeline")
        retrieval_results = {}

        # Initial retrieval
        initial_retrieval_input = self.config["intermediate_files"]["initial_retrieval_input"]
        create_retrieval_input(
            input_files=[self.config["input_file"]],
            output_file=initial_retrieval_input,
            config=self.config,
        )

        initial_retrieval_result = self.config["intermediate_files"]["initial_retrieval_result"]
        run_retrieval(
            input_file=initial_retrieval_input,
            output_file=initial_retrieval_result,
            model_name=self.config["retriever_model"],
            corpus_path=self.config["corpus_path"],
            embeddings_path=self.config["embeddings_path"],
        )

        # Multi-step retrieval
        multi_retrieval_input = self.config["intermediate_files"]["multi_retrieval_input"]
        create_retrieval_input(
            input_files=[self.config["input_file"]],
            output_file=multi_retrieval_input,
            need_retrieval_files=[initial_retrieval_output],
            initial_retrieval_file=initial_retrieval_result,
            multiple_sent=True,
            config=self.config,
        )

        multi_retrieval_result = self.config["intermediate_files"]["multi_retrieval_result"]
        run_retrieval(
            input_file=multi_retrieval_input,
            output_file=multi_retrieval_result,
            model_name=self.config["retriever_model"],
            corpus_path=self.config["corpus_path"],
            embeddings_path=self.config["embeddings_path"],
        )

        retrieval_results["initial"] = {
            "input": initial_retrieval_input,
            "result": initial_retrieval_result,
        }
        retrieval_results["multi"] = {
            "input": multi_retrieval_input,
            "result": multi_retrieval_result,
        }

        return retrieval_results

    def prepare_isrel_issup_data(self, retrieval_results: dict, multi_retrieval_output: str) -> str:
        """Prepare data for isRel and isSup tasks."""
        print("Step 5: Preparing isRel and isSup input data")
        prompt_data_dir = self.config["intermediate_files"]["prompt_data_dir"]

        # Create the output directory if it doesn't exist
        os.makedirs(prompt_data_dir, exist_ok=True)

        # Load the retrieval results
        with open(retrieval_results["multi"]["result"]) as f:
            retrieval_data = json.load(f)

        # Process the data in batches
        batch_size = 100  # Adjust based on your needs
        num_batches = (len(retrieval_data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(retrieval_data))
            batch_data = retrieval_data[start_idx:end_idx]

            # Process this batch (this is a simplified version - you'll need to adapt this
            # to match the actual logic in create_prompt_data.py)
            prompt_data_batch = []
            for item in batch_data:
                # Extract relevant information from the retrieval results
                # and format it according to your needs
                prompt_item = {
                    "instruction": item.get("instruction", ""),
                    "output": item.get("output", ""),
                    "evidence": item.get("evidence", ""),
                    # Add other necessary fields
                }
                prompt_data_batch.append(prompt_item)

            # Save this batch
            batch_output_file = os.path.join(prompt_data_dir, f"prompt_data_batch_{batch_idx:04d}.json")
            with open(batch_output_file, "w") as f:
                json.dump(prompt_data_batch, f, indent=2)

        return prompt_data_dir

    def run_isrel_evaluation(self, prompt_data_dir: str) -> list[str]:
        """Run isRel (relevance) critic evaluation."""
        logging.info("Step 6: Running isRel (relevance) evaluation")
        return self._run_batch_evaluation(
            task="relevance",
            inst_mode="relevance_instruction",
            input_mode="relevance_input",
            prompt_data_dir=prompt_data_dir,
            output_dir_key="rel_eval_dir",
            output_prefix="rel_preds",
            desc="Processing isRel batches",
        )

    def run_issup_evaluation(self, prompt_data_dir: str) -> list[str]:
        """Run isSup (groundness) critic evaluation."""
        logging.info("Step 7: Running isSup (groundness) evaluation")
        return self._run_batch_evaluation(
            task="groundness",
            inst_mode="ground_multi_instruction",
            input_mode="ground_multi_input",
            prompt_data_dir=prompt_data_dir,
            output_dir_key="sup_eval_dir",
            output_prefix="sup_preds",
            desc="Processing isSup batches",
        )

    def combine_all_data(
        self,
        utility_output: str,
        initial_retrieval_output: str,
        multi_retrieval_output: str,
        rel_outputs: list[str],
        sup_outputs: list[str],
        retrieval_results: dict,
    ) -> str:
        """Combine all intermediate results into final training data."""
        logging.info("Step 8: Combining all data into final training set")
        output_dir = self.config["output_dir"]

        final_output = os.path.join(output_dir, "final_training_data.jsonl")

        # Combine all relevance and support predictions
        combined_rel_output = os.path.join(output_dir, "intermediate", "combined_rel_preds.json")
        combined_sup_output = os.path.join(output_dir, "intermediate", "combined_sup_preds.json")

        self._combine_json_files(rel_outputs, combined_rel_output)
        self._combine_json_files(sup_outputs, combined_sup_output)

        # Load all the data
        with open(utility_output) as f:
            utility_data = json.load(f)

        with open(initial_retrieval_output) as f:
            retrieval_i_data = json.load(f)

        with open(multi_retrieval_output) as f:
            retrieval_multi_data = json.load(f)

        with open(combined_sup_output) as f:
            groundness_data = json.load(f)

        with open(combined_rel_output) as f:
            relevance_data = json.load(f)

        with open(self.config["input_file"]) as f:
            orig_input_data = json.load(f)

        with open(retrieval_results["multi"]["result"]) as f:
            retrieval_data = json.load(f)

        final_data = []
        for i in range(len(orig_input_data)):
            combined_item = {
                **orig_input_data[i],
                "utility_pred": (utility_data[i]["pred"] if i < len(utility_data) else ""),
                "retrieval_i_pred": (retrieval_i_data[i]["pred"] if i < len(retrieval_i_data) else ""),
                "retrieval_multi_pred": (retrieval_multi_data[i]["pred"] if i < len(retrieval_multi_data) else ""),
                "groundness_pred": (groundness_data[i]["pred"] if i < len(groundness_data) else ""),
                "relevance_pred": (relevance_data[i]["pred"] if i < len(relevance_data) else ""),
                "retrieval_evidence": (retrieval_data[i].get("evidence", "") if i < len(retrieval_data) else ""),
            }
            final_data.append(combined_item)

        # Save the final output
        os.makedirs(os.path.dirname(os.path.abspath(final_output)), exist_ok=True)
        with open(final_output, "w") as f:
            json.dump(final_data, f, indent=2)

        logging.info(f"Data preparation complete! Final training data saved to: {final_output}")
        return final_output

    def _combine_json_files(self, input_files: list[str], output_file: str):
        """Combine multiple JSON files into one."""
        combined = []
        for f in input_files:
            with open(f) as infile:
                data = json.load(infile)
                if isinstance(data, list):
                    combined.extend(data)
                else:
                    combined.append(data)

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(combined, f, indent=2)

    def run(self):
        """Run the complete data preparation pipeline."""
        logging.info("Starting generator data preparation pipeline...")

        # Step 1-2: Run initial and multi-retrieval critic evaluations
        initial_retrieval_output = self.prepare_initial_retrieval_data()
        multi_retrieval_output = self.prepare_multi_retrieval_data()

        # Step 3: Run utility critic evaluation
        utility_output = self.prepare_utility_data()

        # Step 4: Run retrieval pipeline
        retrieval_results = self.run_retrieval_pipeline(initial_retrieval_output, multi_retrieval_output)

        # Step 5: Prepare isRel and isSup input data
        prompt_data_dir = self.prepare_isrel_issup_data(retrieval_results, multi_retrieval_output)

        # Step 6-7: Run isRel and isSup evaluations
        rel_outputs = self.run_isrel_evaluation(prompt_data_dir)
        sup_outputs = self.run_issup_evaluation(prompt_data_dir)

        # Step 8: Combine all data
        self.combine_all_data(
            utility_output,
            initial_retrieval_output,
            multi_retrieval_output,
            rel_outputs,
            sup_outputs,
            retrieval_results,
        )


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        "input_file",
        "output_dir",
        "critic_model",
        "retriever_model",
        "corpus_path",
        "embeddings_path",
    ]

    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        msg = f"Missing required configuration fields: {', '.join(missing_fields)}"
        raise ValueError(msg)

    return config


if __name__ == "__main__":
    config_path = "generator_data_preparation_config.yaml"
    pipeline = GeneratorDataPreparation(config_path)
    pipeline.run()
