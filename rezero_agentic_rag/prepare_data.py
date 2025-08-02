import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from config import DATA_DIR, logger
from datasets import Dataset, load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load configuration from YAML file
CONFIG_PATH = Path(__file__).parent / "prepare_data_config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Set random seed for reproducibility
random.seed(CONFIG["general"]["random_seed"])


def load_trivia_qa_splits() -> dict[str, Dataset]:
    """Load and preprocess TriviaQA dataset splits."""
    dataset_config = {
        "name": CONFIG["dataset"]["name"],
        "config": CONFIG["dataset"]["config"],
        "max_combined_length": CONFIG["dataset"]["max_combined_length"],
        "splits": CONFIG["dataset"]["splits"],
        "columns_to_remove": CONFIG["dataset"]["columns_to_remove"],
    }
    logger.info(f"Loading and preprocessing {dataset_config['name']} dataset...")

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
            logger.warning(f"Split '{split_key}' not found in dataset. Skipping...")
            continue
        logger.info(f"Processing {split_name} split...")
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
        logger.info(f"  - {split_name}: {len(ds)} examples after filtering")

    return processed_splits


# Load and process dataset
logger.info("Loading dataset splits...")
trivia_splits = load_trivia_qa_splits()

# Verify required splits exist
if "test" not in trivia_splits or "train" not in trivia_splits:
    msg = "Both 'test' and 'train' splits must be present in the dataset"
    raise ValueError(msg)

test_set = trivia_splits["test"]
train_set = trivia_splits["train"]

logger.info(f"\nDataset loaded successfully:\n  - Train examples: {len(train_set)}\n  - Test examples: {len(test_set)}")

# Sample training contexts for corpus
num_contexts = min(CONFIG["sampling"]["num_training_contexts"], len(train_set))
if len(train_set) > num_contexts:
    logger.info(f"Sampling {num_contexts} training contexts...")
    sampled_indices = random.sample(range(len(train_set)), num_contexts)
    train_set = train_set.select(sampled_indices)

# Create document chunks from contexts
logger.info("Creating document chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONFIG["chunking"]["chunk_size"], chunk_overlap=CONFIG["chunking"]["chunk_overlap"]
)

corpus_chunks = []
for i, item in enumerate(train_set):
    # Split each context into chunks
    chunks = text_splitter.split_text(item["context"])
    for chunk in chunks:
        corpus_chunks.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": CONFIG["metadata"]["source_name"],
                    "original_index": i,
                    "original_question": item["question"],
                },
            )
        )

logger.info(f"Created {len(corpus_chunks)} document chunks")

# Save chunks to CSV for inspection
chunks_df = pd.DataFrame(
    {
        "chunk_id": range(1, len(corpus_chunks) + 1),
        "content": [chunk.page_content for chunk in corpus_chunks],
        "metadata": [chunk.metadata for chunk in corpus_chunks],
    }
)
chunks_csv_path = DATA_DIR / "chunks.csv"
chunks_df.to_csv(chunks_csv_path, index=False)
logger.info(f"Saved chunks to {chunks_csv_path}")

# Initialize embeddings
logger.info("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=CONFIG["embedding"]["model_name"],
    model_kwargs=CONFIG["embedding"]["model_kwargs"],
    encode_kwargs=CONFIG["embedding"]["encode_kwargs"],
)

# Create and save FAISS index
logger.info("Creating FAISS index...")
vectorstore = FAISS.from_documents(corpus_chunks, embeddings)

# Save FAISS index
vectorstore.save_local(str(DATA_DIR))
logger.info(f"Saved FAISS index to {DATA_DIR}")

# Create QA pairs from test set
logger.info("Creating QA pairs from test set...")
qa_pairs = []
for item in test_set:
    qa_pairs.append(
        {
            "id": item["question_id"],
            "question": item["question"],
            "answer": item["answer"],
            # No supporting paragraphs since we're using separate corpus
        }
    )

# Save the QA pairs to a JSONL file
questions_path = DATA_DIR / "questions.jsonl"
with open(questions_path, "w") as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair) + "\n")
logger.info(f"Saved {len(qa_pairs)} QA pairs to {questions_path}")
