import json
import pickle
import random
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load configuration
CONFIG_PATH = Path(__file__).parent / "prepare_data_config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Set random seed for reproducibility
random.seed(CONFIG["general"]["random_seed"])


def load_trivia_qa_splits() -> dict[str, Dataset]:
    """Load and preprocess TriviaQA dataset splits.

    Returns:
        Dictionary containing processed dataset splits.
    """
    dataset_config = {
        "name": CONFIG["dataset"]["name"],
        "config": CONFIG["dataset"]["config"],
        "max_combined_length": CONFIG["dataset"]["max_combined_length"],
        "splits": CONFIG["dataset"]["splits"],
        "columns_to_remove": CONFIG["dataset"]["columns_to_remove"],
    }
    print(f"Loading and preprocessing {dataset_config['name']} dataset...")
    # Load the dataset
    dataset = load_dataset(dataset_config["name"], dataset_config["config"])

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        """Flatten context and normalize answers."""
        # Join search contexts into single string
        context_lines = "\n".join(example["search_results"]["search_context"])
        example["context"] = " ".join(context_lines.split("\n"))
        example["answer"] = example["answer"]["normalized_value"]
        return example

    # Process all specified splits
    processed_splits = {}
    for split_name, split_key in dataset_config["splits"].items():
        if split_key not in dataset:
            print(f"Warning: Split '{split_key}' not found in dataset. Skipping...")
            continue
        print(f"Processing {split_name} split...")
        ds = dataset[split_key].map(format_example, num_proc=CONFIG["general"]["num_proc"])
        # Filter empty contexts and limit size
        ds = ds.filter(
            lambda x: (
                len(x["context"]) > 0
                and (len(x["question"]) + len(x["context"])) < dataset_config["max_combined_length"]
            )
        )
        # Remove unnecessary columns
        columns_to_remove = [col for col in dataset_config["columns_to_remove"] if col in ds.column_names]
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)
        processed_splits[split_name] = ds
        print(f"  - {split_name}: {len(ds)} examples after filtering")

    return processed_splits


def save_questions(questions: list[dict[str, str]]) -> Path:
    """Save questions and answers to a JSON file.

    Args:
        questions: List of question-answer dictionaries to save.

    Returns:
        Path to the saved questions file.
    """
    output_config = {
        "data_dir": Path(CONFIG["output"]["data_dir"]),
        "questions_file": CONFIG["output"]["questions_file"],
    }
    output_path = output_config["data_dir"] / output_config["questions_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to {output_path}")
    return output_path


def main() -> None:
    """Main function to prepare the RAG model training data."""
    # Load and process dataset
    print("Loading dataset splits...")
    trivia_splits = load_trivia_qa_splits()
    # Extract test and train splits
    error_msg = "Both 'test' and 'train' splits must be present in the dataset"
    if "test" not in trivia_splits or "train" not in trivia_splits:
        raise ValueError(error_msg)
    test_set = trivia_splits["test"]
    train_set = trivia_splits["train"]

    print(
        f"\nDataset loaded successfully:"
        f"\n  - Train examples: {len(train_set)}"
        f"\n  - Test examples: {len(test_set)}"
    )

    # Create QA pairs from test set
    print("\nCreating QA pairs from test set...")
    qa_pairs = [{"question": item["question"], "answer": item["answer"]} for item in test_set]
    _ = save_questions(qa_pairs)  # Save questions and ignore the returned path

    # Sample training contexts for corpus
    num_contexts = min(CONFIG["sampling"]["num_training_contexts"], len(train_set))
    if len(train_set) > num_contexts:
        print(f"\nSampling {num_contexts} training contexts...")
        train_set = train_set.select(random.sample(range(len(train_set)), num_contexts))

    # Create document chunks from contexts
    print("\nCreating document chunks...")
    corpus_chunks = [
        Document(
            page_content=item["context"],
            metadata={
                "source": CONFIG["metadata"]["source_name"],
                "index": i,
                "question": item["question"],  # Keep original question for reference
            },
        )
        for i, item in enumerate(train_set)
    ]
    print(f"Created {len(corpus_chunks)} document chunks")

    # Initialize embeddings
    print("\nInitializing embeddings...")
    embedding_config = {
        "model_name": CONFIG["embedding"]["model_name"],
        "model_kwargs": CONFIG["embedding"]["model_kwargs"],
        "encode_kwargs": CONFIG["embedding"]["encode_kwargs"],
    }
    embeddings = HuggingFaceEmbeddings(**embedding_config)

    # Create and save FAISS index
    print("\nCreating FAISS index...")
    vectorstore = FAISS.from_documents(corpus_chunks, embeddings)

    # Ensure output directory exists
    output_dir = Path(CONFIG["output"]["faiss_index_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    print(f"Saved FAISS index to '{output_dir}'")

    # Save corpus chunks
    print("\nSaving document chunks...")
    output_dir = Path(CONFIG["output"]["data_dir"])
    chunks_path = output_dir / CONFIG["output"]["chunks_file"]
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "wb") as f:
        pickle.dump(corpus_chunks, f)
    print(f"Saved {len(corpus_chunks)} document chunks to {chunks_path}")
    print("\nData preparation completed successfully!")


if __name__ == "__main__":
    main()
