# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

[Adaptive-RAG](https://arxiv.org/abs/2403.14403) introduces a trained classifier to predict the complexity of query and select the appropriate retrieval strategy based on the complexity level of the query.

The classifier predicts the complexity level of the query and selects the appropriate retrieval strategy based on the query complexity: iterative retrieval for complex queries, single retrieval for moderate queries, and no retrieval for simple queries.

We reproduce the code for training the Adaptive-RAG model based on the implementation provided by the authors at: [https://github.com/starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG).

The code trains a T5 model for classification of query complexity and selects the appropriate retrieval strategy based on the query complexity. We have rewritten the code to use training arguments specified in a YAML file for ease of training.

The dataset used for training is the [Adaptive-RAG training dataset](train.json) released by the authors.

## Project Structure

```bash
adaptive_rag/
   ├── train_adaptive_rag.py
   ├── train_adaptive_rag_config.yaml
   └── requirements.txt
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

The Adaptive-RAG model is trained to predict the complexity level of the query. The training data consists of (Query, Complexity) pairs.

The dataset classifies queries into three complexity levels:  

- **A**: Queries that can be correctly answered by the LLM without any retrieval.  
- **B**: Queries that require at least a single retrieval step to answer correctly.  
- **C**: Queries that require multiple retrieval steps to answer correctly.

The goal in Adaptive-RAG is to use the simplest, most effective approach for each query. By identifying the complexity level of the query, we can select the appropriate retrieval strategy.

To run the training script, use the following command:

```bash
python train_adaptive_rag.py
```

The training script takes a YAML configuration file as input. The training parameters can be specified in `train_adaptive_rag_config.yaml`.