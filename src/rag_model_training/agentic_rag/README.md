# Agentic RAG: Training Language Models as Autonomous Agents with Retrieval Capabilities

Agentic RAG implements a training approach for language models that enables them to act as autonomous agents with retrieval capabilities. The model is trained using Group Relative Policy Optimization (GRPO) to recognize missing information, rewrite queries, and generate tool calls for retrieval autonomously.

This implementation is based on the [AutoDidact](https://github.com/dCaples/AutoDidact) framework and applies it to the TriviaQA dataset, training a Llama-3-8B model to function as an autonomous agent that can search for information when needed.

## Project Structure

```bash
agentic_rag/
├── README.md
├── __init__.py
├── prepare_data.py
├── prepare_data_config.yaml
├── train.py
├── training_config.yaml
├── inference.py
├── inference_config.yaml
├── evaluation.py
├── search_module.py
├── prompts.py
├── rl_helpers.py
├── rewards.py
├── unsloth_grpo_trainer_agent.py
└── requirements.txt
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Approach

Agentic RAG trains language models to function as autonomous agents with the following capabilities:

1. **Self-Recognition of Missing Information**: The model can identify when it lacks sufficient knowledge to answer a query accurately.

2. **Query Rewriting and Refinement**: The model can autonomously rewrite or refine queries to improve retrieval effectiveness.

3. **Tool Call Generation**: The model generates appropriate tool calls (specifically search queries) to retrieve relevant information from a knowledge corpus.

4. **LLM-as-a-Judge Reward Function**: A correctness reward function evaluates whether the model makes appropriate tool calls and retrieves relevant information.

## Key Features

- **Fine-tuned Llama-3-8B model** on the TriviaQA dataset using GRPO
- **Autonomous agent capabilities** for recognizing missing information and generating retrieval tool calls
- **LoRA-based parameter-efficient training** for reduced memory usage
- **FAISS-based retrieval system** for efficient document search
- **LLM-as-a-Judge correctness reward function** for training signal

## Training

The training process uses Group Relative Policy Optimization (GRPO) with the following key components:

### 1. Data Preparation

Prepare the training data by running:

```bash
python prepare_data.py
```

This script:

- Loads the TriviaQA dataset
- Processes questions and answers
- Creates a searchable corpus from document contexts
- Builds a FAISS index for efficient retrieval
- Saves processed data in the `saved_data/` directory

Configuration options can be adjusted in `prepare_data_config.yaml`.

### 2. Training the Agentic Model

Train the model using GRPO:

```bash
python train.py
```

Training features:

- Fine-tunes Llama-3-8B using GRPO
- Uses LoRA for parameter-efficient training
- Implements agentic generation with autonomous search capabilities
- Applies correctness-based reward functions
- Integrates with vLLM for efficient generation

Key training parameters can be configured in `training_config.yaml`:

- Model configuration (base model, sequence length, quantization)
- LoRA parameters (rank, alpha, target modules)
- Training hyperparameters (learning rate, batch size, steps)
- Sampling parameters for generation
- Agent configuration (max generations)

### 3. Reward Functions

The training uses two reward functions:

1. **Correctness Reward**: Evaluates whether the final answer is correct using an LLM-as-a-Judge approach
2. **Formatting Reward**: Rewards proper tool call formatting and penalizes errors

### 4. Inference

Run inference with the trained model:

```bash
python inference.py
```

The inference script loads the trained model and runs evaluation on test data. Configuration options can be adjusted in `inference_config.yaml`.

### 5. Evaluation

The evaluation process measures the model's performance on test data, calculating the percentage of correct answers.

```bash
python inference.py
```

The evaluation script:

- Loads the trained model with LoRA adapters using the configuration from `inference_config.yaml`
- Loads the test dataset from the TriviaQA dataset
- Runs the agentic generation process on test questions using the `run_agent` function
- Uses an LLM-as-a-Judge approach to verify answer correctness
- Calculates and reports the percentage of correct answers

The evaluation process works as follows:

1. **Model Loading**: The script loads the base Llama-3.1-8B model with LoRA adapters trained during the GRPO process
2. **Dataset Loading**: Test questions are loaded from the TriviaQA dataset.
3. **Agentic Generation**: For each question, the model goes through an agentic process:
   - The agent can perform up to 6 generations (configurable)
   - It can make tool calls to search for information when needed
   - The agent finishes when it has answered the question or reached the maximum generations
4. **Answer Verification**: Uses a reward function to verify answer correctness:
   - Extracts the final answer from the agent's conversation
   - Uses a separate LLM to judge whether the final answer is correct compared to the ground truth
5. **Results Calculation**: Calculates the percentage of correct answers and prints the results

Configuration options can be adjusted in `inference_config.yaml`:

- Model configuration (base model, sequence length, quantization)
- LoRA parameters for loading adapters
- Sampling parameters for generation (temperature, top_p, max_tokens)
- Agent configuration (max generations)

The evaluation results are printed showing the percentage of correct answers.
