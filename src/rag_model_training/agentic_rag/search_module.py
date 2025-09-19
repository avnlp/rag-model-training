# This code is based on the implementation from: https://github.com/dCaples/AutoDidact/blob/main/search_module.py.

import json
import pickle
import random

from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "intfloat/multilingual-e5-large"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": False}
TEST_SIZE = 0.1
SEED = 42


def load_vectorstore():
    """Load the pre-saved FAISS index.

    Returns:
        FAISS: Loaded FAISS vectorstore object or None if failed.
    """
    try:
        import os

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME, model_kwargs=MODEL_KWARGS, encode_kwargs=ENCODE_KWARGS
        )

        # Load the FAISS index with absolute path
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
        print(f"Loading FAISS index from: {index_path}")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Successfully loaded FAISS index")
        return vectorstore

    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        import traceback

        traceback.print_exc()
        return None


# Load vectorstore when module is imported
try:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Warning: FAISS vectorstore could not be loaded.")
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    vectorstore = None


def search(query: str, return_type=str, results: int = 5) -> str | list[str]:
    """Search for relevant chunks using similarity search.

    Args:
        query: The search query
        return_type: Return as string or list (default: str)
        results: Number of results to return (default: 5)

    Returns:
        Results as string or list depending on return_type

    Raises:
        ValueError: If vectorstore is not loaded or invalid return_type is provided
    """
    if vectorstore is None:
        msg = "Vectorstore not loaded. Please ensure FAISS index exists."
        raise ValueError(msg)

    search_results = vectorstore.similarity_search(query, k=results)

    if return_type is str:
        str_results = ""
        for idx, result in enumerate(search_results, start=1):
            str_results += f"Result {idx}:\n"
            str_results += result.page_content + "\n"
            str_results += "------\n"
        return str_results
    elif return_type is list:
        return [result.page_content for result in search_results]
    else:
        msg = "Invalid return_type. Use str or list."
        raise ValueError(msg)


def load_qa_data():
    """Load the pre-generated questions and document chunks.

    Returns:
        tuple: A tuple containing (chunks, questions) or (None, None) if failed.
    """
    try:
        import os

        # Get absolute paths to data files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chunks_path = os.path.join(base_dir, "saved_data", "chunks.pkl")
        questions_path = os.path.join(base_dir, "saved_data", "questions.json")

        print(f"Loading chunks from: {chunks_path}")
        print(f"Loading questions from: {questions_path}")

        # Load the chunks
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # Load the questions
        with open(questions_path) as f:
            questions = json.load(f)

        print(f"Successfully loaded {len(chunks)} chunks and {len(questions)} questions")
        return chunks, questions
    except Exception as e:
        print(f"Error loading QA data: {e}")
        import traceback

        traceback.print_exc()
        return None, None


# Load chunks and questions when module is imported
try:
    chunks, questions = load_qa_data()
    if chunks is None or questions is None:
        print("Warning: Could not load QA data.")
except Exception as e:
    print(f"Error initializing QA data: {e}")
    chunks, questions = None, None


def get_question_answer(idx: int | None = None, return_both: bool = True) -> dict | str:
    """Get a question-answer pair either by index or randomly.

    Args:
        idx: Index of the question to retrieve (if None, selects random question)
        return_both: Whether to return both question and answer (default: True)

    Returns:
        dict | str: Question and answer as dict if return_both=True, otherwise just the question

    Raises:
        ValueError: If questions are not loaded or index is out of range
    """
    if questions is None:
        msg = "Questions not loaded. Please ensure questions.json exists."
        raise ValueError(msg)

    if idx is None:
        # Select a random question
        qa_pair = random.choice(questions)
    elif 0 <= idx < len(questions):
        # Select question by index
        qa_pair = questions[idx]
    else:
        msg = f"Index out of range. Must be between 0 and {len(questions)-1}"
        raise ValueError(msg)

    question = qa_pair["question"]
    answer = qa_pair["answer"]

    if return_both:
        return {"question": question, "answer": answer}
    else:
        return question


def get_question_count() -> int:
    """Get the total number of available questions.

    Returns:
        int: Total number of questions

    Raises:
        ValueError: If questions are not loaded
    """
    if questions is None:
        msg = "Questions not loaded. Please ensure questions.json exists."
        raise ValueError(msg)
    return len(questions)


def get_qa_dataset():
    """Return a HuggingFace Dataset containing question and answer pairs.

    This dataset is constructed from the loaded questions data (questions.json).
    Each element in the dataset is a dictionary that includes at least:
      - "question": The question text.
      - "answer": The corresponding answer text.
    Additional keys present in the original questions data will also be included.

    Returns:
        tuple: A tuple of (train_dataset, test_dataset) split from the full dataset.
    """
    if questions is None:
        msg = "Questions not loaded. Please ensure questions.json exists."
        raise ValueError(msg)

    qa_dataset = Dataset.from_list(questions)
    full_dataset = qa_dataset.shuffle(seed=SEED)
    train_dataset = full_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)["train"]
    test_dataset = full_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)["test"]
    train_dataset = train_dataset.rename_column("question", "prompt")
    test_dataset = test_dataset.rename_column("question", "prompt")

    return train_dataset, test_dataset
