# RAG Model Training

This repository provides training code for various advanced Retrieval-Augmented Generation (RAG) techniques. The repository focuses on training models for four main approaches: 
- **Adaptive-RAG**, which trains a classifier to predict query complexity and optimize retrieval strategies.
- **Corrective RAG**, which trains models to evaluate and score document relevance.
- **RQ-RAG**, which trains models for query refinement through rewriting, decomposition, and disambiguation.
- **Self-RAG**, which trains models for self-reflection on retrieval decisions and generation quality. 

## Adaptive-RAG (Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity)

Adaptive-RAG introduces a trained classifier to predict query complexity and select the appropriate retrieval strategy:

- **Simple Queries (A)**: Queries that can be correctly answered by the LLM without any retrieval.

- **Moderate Queries (B)**: Queries that require at least a single retrieval step to answer correctly.

- **Complex Queries (C)**: Queries that require multiple retrieval steps to answer correctly.

This approach optimizes computational resources by using the simplest effective strategy for each query.

The code trains a T5 model for classification of query complexity and selects the appropriate retrieval strategy based on the query complexity.

More details on the Adaptive-RAG training can be found [here](adaptive_rag/README.md).

## Corrective RAG (Corrective Retrieval Augmented Generation)
Corrective RAG enhances RAG by introducing a Retrieval Evaluator that:
- Classifies retrieved documents as Correct, Ambiguous, or Incorrect
- Uses a scoring mechanism (-1 to 1) to evaluate document relevance
- Improves document utilization in the RAG pipeline

This approach helps in identifying and handling low-quality or irrelevant retrievals.

The code trains a T5 model for retrieval evaluation on the training dataset provided by the authors. 

More details on the Corrective RAG training can be found [here](corrective_rag/README.md).

## RQ-RAG (Learning to Refine Queries for Retrieval-Augmented Generation)
RQ-RAG enhances RAG with explicit query refinement capabilities:
- **Query Rewriting**: Improves the original query
- **Query Decomposition**: Breaks down complex queries into simpler ones
- **Query Disambiguation**: Resolves ambiguous queries
The model uses special tokens to guide the generation process and follows a tree-based decoding strategy for iterative refinement.

The code trains the RQ-RAG classifier on the Llama-3.2 model. 
More details on the RQ-RAG training can be found [here](rq_rag/README.md).

## Self-RAG (Self-Reflective Retrieval-Augmented Generation)

Self-RAG introduces a framework for self-reflection during RAG using special tokens that evaluate:

- **Retrieval Decision**: Whether to retrieve documents
- **Relevance**: Quality of retrieved documents
- **Grounding**: Whether the generation is supported by evidence
- **Utility**: Overall quality of the generation

The approach involves training both a critic model and a generator model in two phases.
We have trained a T5 model for Self-RAG on the Earnings Calls Dataset.

More details on the Self-RAG training can be found [here](self_rag/README.md).
