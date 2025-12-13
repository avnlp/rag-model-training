# ReZero Agentic RAG: Training Language Models as Autonomous Agents with Retrieval Capabilities

ReZero Agentic RAG implements a training approach for language models that enables them to act as autonomous agents with retrieval capabilities. The model is trained using Group Relative Policy Optimization (GRPO) to recognize missing information, rewrite queries, and generate tool calls for retrieval autonomously.

This implementation is based on the [ReZero](https://arxiv.org/abs/2504.1001) framework and applies it to the TriviaQA dataset, training a Llama-3.2-8B model to function as an autonomous agent that can search for information when needed.

## Project Structure

```bash
rezero_agentic_rag/
├── README.md
├── __init__.py
├── agent.py
├── config.py
├── evaluation.py
├── prepare_data.py
├── prepare_data_config.yaml
├── prompts.py
├── rewards.py
├── search_module.py
├── tokenizer_adapter.py
├── train.py
├── unsloth_grpo_trainer_agent.py
└── requirements.txt
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Approach

ReZero Agentic RAG trains language models to function as autonomous agents with the following capabilities:

1. **Self-Recognition of Missing Information**: The model can identify when it lacks sufficient knowledge to answer a query accurately.

2. **Query Rewriting and Refinement**: The model can autonomously rewrite or refine queries to improve retrieval effectiveness.

3. **Tool Call Generation**: The model generates appropriate tool calls (specifically search queries) to retrieve relevant information from a knowledge corpus.

4. **LLM-as-a-Judge Reward Function**: A correctness reward function evaluates whether the model makes appropriate tool calls and retrieves relevant information.

5. **Search Strategy Optimization**: The model learns effective search strategies, including how to process retrieved information and refine subsequent searches.

6. **Search Retry and Query Refinement**: The model learns to retry searches when initial attempts are insufficient, refining queries based on available information.

## Key Features

- **Fine-tuned Llama-3.2-8B model** on the TriviaQA dataset using GRPO.
- **Autonomous agent capabilities** for recognizing missing information and generating retrieval tool calls.
- **LoRA-based parameter-efficient training** for reduced memory usage.
- **FAISS-based retrieval system** for efficient document search.
- **Composite LLM-as-a-Judge reward function** for better training signal.
- **Multiple reward functions** including correctness, format, retry, search diversity, and search strategy.
- **Agentic generation with multiple search-refinement cycles** to improve answer quality.

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
- Builds a FAISS index for efficient retrieval using multilingual-e5-large embeddings
- Saves processed data in the `data/` directory
- Creates chunks.csv for inspection of document chunks

Configuration options can be adjusted in `prepare_data_config.yaml`.

### 2. Training the Agentic Model

Train the model using GRPO:

```bash
python train.py
```

Training features:

- Fine-tunes Llama-3.2-8B using GRPO
- Uses LoRA for parameter-efficient training (rank 64)
- Implements agentic generation with autonomous search capabilities
- Applies composite reward functions
- Integrates with vLLM for efficient generation

Key training parameters can be configured in the config.py file:

- Model configuration (base model, sequence length, quantization)
- LoRA parameters (rank 64, target modules)
- Training hyperparameters (learning rate 5e-6, batch size, steps)
- Sampling parameters for generation
- Agent configuration (max generations up to 32)

### 3. Reward Functions

The training uses six reward functions:

1. **Correctness Reward**: Evaluates whether the final answer is correct using an LLM-as-a-Judge approach
2. **Formatting Reward**: Rewards proper tag usage ('think', 'search', 'answer') and penalizes errors
3. **Retry Reward**: Encourages optimal retry behavior with search attempts, capped at optimal count
4. **EM Chunk Reward**: Checks if model's search results contain the last necessary supporting paragraph
5. **Search Strategy Reward**: Evaluates good search strategy and query analysis steps
6. **Search Diversity Reward**: Evaluates diversity of search queries in a conversation

### 4. Agentic Generation Process

The model implements an iterative search-and-reason process:

- The agent starts with a question and uses 'think' tags to reason about the information needed
- If information is lacking, the agent generates 'search' queries to retrieve relevant information
- The agent processes retrieved information and continues the reasoning process
- When confident in the answer, the agent provides the final response in 'answer' tags
- The agent can perform multiple search-refinement cycles to improve answer quality

### 5. Evaluation

The evaluation process measures the model's performance on test data, calculating the percentage of correct answers.

```bash
python evaluation.py
```

The evaluation script:

- Loads the trained model with LoRA adapters
- Loads the test dataset from the TriviaQA dataset
- Runs the agentic generation process on test questions using the `run_agent` function
- Uses an LLM-as-a-Judge approach to verify answer correctness
- Calculates and reports the percentage of correct answers

The evaluation process works as follows:

1. **Model Loading**: The script loads the base Llama-3.2-8B model with LoRA adapters trained during the GRPO process
2. **Dataset Loading**: Test questions are loaded from the TriviaQA dataset.
3. **Agentic Generation**: For each question, the model goes through an agentic process:
   - The agent can perform up to 32 generations (configurable)
   - It can make tool calls to search for information when needed
   - The agent finishes when it has answered the question or reached the maximum generations
4. **Answer Verification**: Uses a reward function to verify answer correctness:
   - Extracts the final answer from the agent's conversation
   - Uses a separate LLM to judge whether the final answer is correct compared to the ground truth
5. **Results Calculation**: Calculates the percentage of correct answers and prints the results

## Configuration

The system uses multiple configuration files:

- `prepare_data_config.yaml`: Configures data preparation parameters including dataset selection, chunking settings, and embedding models
- `config.py`: Contains model parameters, training hyperparameters, and path configurations
- Training parameters are set in the `TRAINING_CONFIG` dictionary with options for learning rate, batch size, and other hyperparameters
