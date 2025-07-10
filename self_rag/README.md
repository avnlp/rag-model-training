# Self-RAG for Earnings Call Analysis

Self-RAG introduces a method for Self-Reflection during RAG by the use of special tokens for checking retrieval, relevance, grounding and utility of the retrieved documents. This technique improves the generation quality of the RAG system.

This repository contains an implementation of Self-RAG (Retrieval-Augmented Generation) specifically adapted for analyzing Earnings Call data.

The implementation is based on the [SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection](https://arxiv.org/abs/2310.11511) paper. The code is based on the repository: [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag).

This implementation adapts the Self-RAG framework specifically for Earnings Call analysis, with the following key modifications:

- **Custom Data Processing**: Specialized processing of Earnings Call transcripts.
- **Modular Architecture**: Separate components for critic and generator training.
- **Flexible Configuration**: YAML-based configuration for all components.
- **Efficient Training**: Support for LoRA (Low-Rank Adaptation) to reduce memory usage.

## Project Structure

```bash
self_rag/
├── README.md
├── requirements.txt
│
├── Earnings Call Data Processing
│   │   # Processes raw earnings call data
│   └── process_earnings_call_data.py       
│
├── Critic Training
│   │   # Creates critic training data with [RETRIEVAL] tokens
│   ├── critic_retrieval_collector.py 
│   │   # Creates critic training data with [RELEVANT]/[IRRELEVANT] tokens
│   ├── critic_relevance_collector.py       
│   │   # Creates critic training data with [UTILITY:1]/[UTILITY:2].../[UTILITY:5] tokens
│   ├── critic_utility_collector.py         
│   │   # Creates critic training data with [SUPPORTED]/[PARTIALLY_SUPPORTED]/[NOT_SUPPORTED] tokens
│   ├── critic_groundness_collector.py      
│   │   # Combines all critic training data into a single JSON file
│   ├── critic_data_processor.py            
│   │   # Trains the critic model on the critic training data with special tokens
│   └── train_critic.py                    
│
└── Generator Training
    │   # Prepares generator training data by running the retriever and evaluating the retrieved documents using the critic model
    ├── generator_data_preparation.py       
    │   # Trains the generator model on the generator training data with special tokens for retrieval, relevance, grounding and utility
    └── train_generator.py                  
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

The training is carried out in two phases:

- Phase 1: Train a critic model to evaluate retrieval and generation quality.
- Phase 2: Train a generator model using the critic's feedback to generate responses.

The parameters for the data preparation and training are specified in the corresponding YAML configuration files.

### 1. Process Earnings Call Data

The Earnings Call Data is processed to the following format:

```python
{
    "id": str # unique instance id,
    "instruction": str, # input instruction
    "input": str # input question
    "evidence": str, # context for the answer
    "output": str, # answer
}
```

```bash
python process_earnings_call_data.py
```

### 2. Generate Critic Training Data

To train the Critic Model, we need to generate training data with special tokens for retrieval, relevance, grounding and utility. Each file processes the Earnings Call Data and adds the special tokens to the data.

For retrieval we use the `[Retrieval]`/`[No Retrieval]` tokens to indicate whether the model should retrieve documents from the context.

```bash
python critic_retrieval_collector.py
```

For relevance we use the `[Relevant]`/`[Irrelevant]` tokens to indicate whether the retrieved documents are relevant to the question.

```bash
python critic_relevance_collector.py
```

For grounding we use the `[Grounded]`/`[Not Grounded]` tokens to indicate whether the retrieved documents are grounded in the question.

```bash
python critic_groundness_collector.py
```

For utility we use the `[Utility:1]`/`[Utility:2]`/`[Utility:3]`/`[Utility:4]`/`[Utility:5]` tokens to indicate the utility of the retrieved documents.

```bash
python critic_utility_collector.py
```

We combine all the critic data for each of the special tokens into a single JSON file along with the original question and answer. The full prompt including the instruction and input is also included.

```bash
python critic_data_processor.py
```

### 3. Train Critic Model

The critic model is trained to evaluate different aspects of the generation and accordingly generate a special token for each aspect:

1. **Retrieval Decision**: Whether retrieval is needed. (`[Retrieval]`/`[No Retrieval]`)
2. **Relevance**: Whether retrieved documents are relevant. (`[Relevant]`/`[Irrelevant]`)
3. **Grounding**: Whether the generation is grounded in the retrieved documents. (`[Grounded]`/`[Not Grounded]`)
4. **Utility**: Overall quality of the generation. (`[Utility:1]`/`[Utility:2]`/`[Utility:3]`/`[Utility:4]`/`[Utility:5]`)

```bash
python train_critic.py
```

### 4. Prepare Generator Training Data

The data creation for Generator consists of the following steps:

1. Run Critic to judge retrieval tokens based on input question and evidence.
2. Run Critic to to judge isUse (Utility) based on question.
3. Run initial retriever and get retrieved documents.
4. Evaluate isRel (Relevance) and isSUP (Grounding) of the retrieved documents.
5. Generate final output based on retrieved documents and decisions with special tokens from Critic.

```bash
python generator_data_preparation.py
```

### 5. Train Generator Model

The generator model is trained to generate text while making retrieval decisions based on the critic's feedback.

```bash
python train_generator.py
```
