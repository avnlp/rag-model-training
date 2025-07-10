"""
Retrieval Utilities for Generator Data Preparation

This module provides utility functions and classes for handling retrieval operations
in the generator data preparation pipeline, including document retrieval and result processing.

This code is based on the implementation from: https://github.com/AkariAsai/self-rag/tree/main/retrieval_lm/
"""

import glob
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, TypedDict

import jsonlines
import numpy as np
import src.contriever
import src.data
import src.index
import src.normalize_text
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Retriever:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.index = None
        self.passages = None
        self.passage_id_map = None

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if (
                    len(batch_question) == args.per_gpu_batch_size
                    or k == len(queries) - 1
                ):
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = (
                np.vstack((allembeddings, embeddings))
                if allembeddings.size
                else embeddings
            )
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size
                )

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def setup_retriever(self):
        print(f"Loading model from: {self.args.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(
            self.args.model_name_or_path
        )
        self.model.eval()
        self.model = self.model.cuda()
        if not self.args.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(
            self.args.projection_size, self.args.n_subquantizers, self.args.n_bits
        )

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(
                self.index, input_paths, self.args.indexing_batch_size
            )
            print(f"Indexing time: {time.time() - start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(
            questions_embedding, self.args.n_docs
        )
        print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]

    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(
        self,
        model_name_or_path,
        passages,
        passages_embeddings,
        save_or_load_index=False,
    ):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(
            model_name_or_path
        )
        self.model.eval()
        self.model = self.model.cuda()

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time() - start_time_indexing:.1f} s.")

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


def create_retrieval_input(
    input_files: List[str],
    output_file: str,
    need_retrieval_files: Optional[List[str]] = None,
) -> None:
    """
    Create input data for retrieval by combining data from multiple input files.

    Args:
        input_files: List of input JSON/JSONL files
        output_file: Path to save the output file
        need_retrieval_files: Optional list of files containing retrieval necessity predictions
    """
    combined_data = []
    seen_ids = set()

    # Load data from all input files
    for input_file in input_files:
        if input_file.endswith(".json"):
            with open(input_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
            with jsonlines.open(input_file, "r") as reader:
                for item in reader:
                    data.append(item)

        # Add data to combined list, avoiding duplicates by ID
        for item in data:
            item_id = item.get("id")
            if item_id is None or item_id not in seen_ids:
                combined_data.append(item)
                if item_id is not None:
                    seen_ids.add(item_id)

    # Apply retrieval necessity filter if needed
    if need_retrieval_files:
        need_retrieval: set[str] = set()
        for retrieval_file in need_retrieval_files:
            if retrieval_file.endswith(".json"):
                with open(retrieval_file, "r") as f:
                    retrieval_data = json.load(f)
                    if isinstance(retrieval_data, dict):
                        need_retrieval.update(retrieval_data.keys())
                    else:
                        need_retrieval.update(str(item) for item in retrieval_data)
            else:
                with open(retrieval_file, "r") as f:
                    need_retrieval.update(line.strip() for line in f if line.strip())

        combined_data = [
            item for item in combined_data if str(item.get("id", "")) in need_retrieval
        ]

    # Save combined data
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output_file.endswith(".json"):
        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=2)
    else:
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(combined_data)


class RetrieverArgs:
    def __init__(self, model_name, corpus_path, embeddings_path, n_docs):
        self.model_name_or_path = model_name
        self.passages = corpus_path
        self.passages_embeddings = embeddings_path
        self.n_docs = n_docs
        self.projection_size = 768
        self.n_subquantizers = 0
        self.n_bits = 8
        self.no_fp16 = False
        self.save_or_load_index = True
        self.indexing_batch_size = 1000000
        self.lowercase = True
        self.normalize_text = True
        self.question_maxlength = 200
        self.per_gpu_batch_size = 16


def run_retrieval(
    input_file: str,
    output_file: str,
    model_name: str,
    corpus_path: str,
    embeddings_path: str,
    n_docs: int = 10,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run retrieval using the specified model.

    Args:
        input_file: Path to input JSON/JSONL file
        output_file: Path to save retrieval results
        model_name: Name/path of the retrieval model
        corpus_path: Path to the corpus for retrieval
        embeddings_path: Path to the embeddings for retrieval
        n_docs: Number of documents to retrieve per query
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # If config is provided, ensure all paths are expanded
    if config:
        output_file = os.path.expandvars(output_file)
        model_name = os.path.expandvars(model_name)
        corpus_path = os.path.expandvars(corpus_path)
        embeddings_path = os.path.expandvars(embeddings_path)
        n_docs = config.get("n_docs", n_docs)

    # Load input data
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            data = json.load(f)
    else:
        data = []
        with jsonlines.open(input_file, "r") as reader:
            for item in reader:
                data.append(item)

    # Set up retriever
    args = RetrieverArgs(model_name, corpus_path, embeddings_path, n_docs)
    retriever = Retriever(args)
    retriever.setup_retriever_demo(
        model_name_or_path=model_name,
        passages=corpus_path,
        passages_embeddings=embeddings_path,
        n_docs=n_docs,
        save_or_load_index=True,
    )

    # Process each query
    results: list[dict] = []
    for item in data:
        query = item.get("question", "")
        query_id = item.get("id", str(len(results)))

        # Get top documents
        top_docs = retriever.search_document_demo(query, n_docs=n_docs)

        # Format results
        result = {
            "id": query_id,
            "question": query,
            "docs": [doc["text"] for doc in top_docs],
            "scores": [1.0] * len(top_docs),
        }
        results.append(result)

    # Save results
    with open(output_file, "w") as f:
        if output_file.endswith(".json"):
            json.dump(results, f, indent=2)
        else:
            writer = jsonlines.Writer(f)
            writer.write_all(results)
            writer.close()


class RetrievalResult(TypedDict):
    docs: List[str]
    scores: List[float]


def process_retrieval_results(
    input_file: str, output_file: str, top_k: int = 5
) -> Dict[str, RetrievalResult]:
    """
    Process retrieval results and extract top-k documents.

    Args:
        input_file: Path to retrieval results file
        output_file: Path to save processed results
        top_k: Number of top documents to keep

    Returns:
        Dictionary containing processed retrieval results
    """
    # Load retrieval results
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            results = json.load(f)
    else:
        results = []
        with jsonlines.open(input_file, "r") as reader:
            for obj in reader:
                results.append(obj)

    # Process results to keep only top-k documents
    processed_results: Dict[str, RetrievalResult] = {}
    for item in results:
        query_id = item.get("id", str(len(processed_results)))
        docs = item.get("docs", [])
        scores = item.get("scores", [])

        # Sort by score (descending) and keep top-k
        sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        processed_results[query_id] = {
            "docs": [doc for doc, _ in sorted_docs],
            "scores": [float(score) for _, score in sorted_docs],
        }

    # Save processed results
    with open(output_file, "w") as f:
        json.dump(processed_results, f, indent=2)

    return processed_results


def load_retrieval_results(file_path: str) -> Dict[str, Any]:
    """
    Load retrieval results from a file.

    Args:
        file_path: Path to the retrieval results file

    Returns:
        Dictionary containing the loaded results
    """
    with open(file_path, "r") as f:
        return json.load(f)


def merge_retrieval_results(
    results_list: List[Dict[str, RetrievalResult]], output_file: Optional[str] = None
) -> Dict[str, RetrievalResult]:
    """
    Merge multiple retrieval results.

    Args:
        results_list: List of retrieval result dictionaries
        output_file: Optional path to save merged results

    Returns:
        Dictionary containing merged results
    """
    merged_results: Dict[str, RetrievalResult] = {}

    for results in results_list:
        for query_id, result in results.items():
            if query_id not in merged_results:
                merged_results[query_id] = RetrievalResult(docs=[], scores=[])

            merged_results[query_id]["docs"].extend(result["docs"])
            merged_results[query_id]["scores"].extend(result["scores"])

    # Sort by score and remove duplicates
    for query_id in merged_results:
        # Create a set to track seen documents
        seen_docs = set()
        unique_docs = []
        unique_scores = []

        # Sort by score (descending)
        sorted_items = sorted(
            zip(merged_results[query_id]["docs"], merged_results[query_id]["scores"]),
            key=lambda x: x[1],
            reverse=True,
        )

        # Keep only unique documents
        for doc, score in sorted_items:
            if doc not in seen_docs:
                seen_docs.add(doc)
                unique_docs.append(doc)
                unique_scores.append(score)

        merged_results[query_id] = {"docs": unique_docs, "scores": unique_scores}

    # Save to file if output path is provided
    if output_file:
        with open(output_file, "w") as f:
            json.dump(merged_results, f, indent=2)

    return merged_results
