# Corrective Retrieval Augmented Generation

[Corrective RAG](https://arxiv.org/abs/2401.15884) is a technique which introduces a Retrieval Evaluator to self-correct the results of the retriever and improve the utilization of documents in RAG pipelines.

We reproduce the code for training the Corrective RAG model based on the implementation provided by the authors at: [https://github.com/HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG).

The code trains a T5 model for retrieval evaluation on the training dataset provided by the authors. We have rewritten the code to use training arguments specified in a YAML file for ease of training.

The dataset used for training is the [CRAG training data](train_data.txt) released by the authors.

## Project Structure

```bash
corrective_rag/
   ├── train_corrective_rag.py
   ├── train_corrective_rag_config.yaml
   └── requirements.txt
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

- The corrective rag evaluator model is trained on (Question + Retrieved Documents, Label) pairs. The model is trained to output a score that classifies the retrieved documents into Correct, Ambiguous and Incorrect based on the question.
- The classifier model scores the document based on the question passed with it. The score is a float ranging from -1 to 1.
- The label of positive samples was 1, while that of negative ones was -1. Two thresholds were set to classify the retrieved text into Correct, Ambiguous and Incorrect.

To run the training script, use the following command:

```bash
python train_corrective_rag.py
```

The training script takes a YAML configuration file as input. The training parameters can be specified in `train_corrective_rag_config.yaml`.
