<h1 align="center">RAG Model Training</h1>

<div align="center">

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avnlp/rag-model-training)
[![CI](https://img.shields.io/github/actions/workflow/status/avnlp/rag-model-training/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/avnlp/rag-model-training/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/github/actions/workflow/status/avnlp/rag-model-training/ci.yml?branch=main&label=Ruff&logo=ruff)](https://github.com/avnlp/rag-model-training/actions/workflows/ci.yml)
[![MyPy](https://img.shields.io/github/actions/workflow/status/avnlp/rag-model-training/ci.yml?branch=main&label=MyPy&logo=python)](https://github.com/avnlp/rag-model-training/actions/workflows/ci.yml)
[![Bandit](https://img.shields.io/github/actions/workflow/status/avnlp/rag-model-training/ci.yml?branch=main&label=Bandit&logo=owasp)](https://github.com/avnlp/rag-model-training/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/avnlp/rag-model-training?color=green)](https://github.com/avnlp/rag-model-training/blob/main/LICENSE)

</div>

This repository provides code for training LLMs for various advanced Retrieval-Augmented Generation (RAG) techniques. Each approach targets a different component of the RAG pipeline: retrieval optimization, joint retrieval and generation evaluation, or agentic generation with autonomous retrieval.

The implementations reproduce the original training methodologies released by the authors of the papers and apply them across different domains and datasets.

| Technique | Description | Paper | Training Approach |
|-----------|-------------|-------|------------------|
| **Adaptive-RAG** | Trains a classifier to predict query complexity and optimize retrieval strategies | [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) | SFT |
| **Corrective RAG** | Trains models to evaluate and score document relevance | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | SFT |
| **RQ-RAG** | Trains models for query refinement through rewriting, decomposition, and disambiguation | [Learning to Refine Queries for Retrieval Augmented Generation](https://arxiv.org/abs/2404.00610) | SFT |
| **Self-RAG** | Trains models for self-reflection on retrieval decisions and generation quality | [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | Two-phase SFT |
| **Agentic RAG** | Trains language models as autonomous agents with retrieval capabilities | [AutoDidact](https://github.com/dCaples/AutoDidact) | GRPO |
| **ReZero Agentic RAG** | Implements agent-based RAG with search and evaluation modules | [ReZero: Enhancing LLM search ability by trying one-more-time](https://arxiv.org/abs/2504.11001) | GRPO |

## Adaptive-RAG

**Paper**: [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)

Adaptive-RAG introduces a trained classifier to predict query complexity and select the appropriate retrieval strategy. The model learns to classify queries into three types: Simple (answerable without retrieval), Moderate (requiring single retrieval), and Complex (requiring multiple retrievals). This approach optimizes computational resources by using the simplest effective strategy for each query.

- **Model**: T5-large classifier for query complexity prediction
- **Dataset**: [Adaptive-RAG training dataset](https://github.com/starsuzi/Adaptive-RAG) (from Musique and other multi-hop QA datasets)
- **Training Approach**: SFT - reproduces original paper's training methodology exactly
- **Key Innovation**: Dynamically selects retrieval depth based on query complexity

For configuration and training setup, see the [Adaptive-RAG documentation](src/rag_model_training/adaptive_rag/README.md).

## Corrective RAG

**Paper**: [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)

Corrective RAG enhances RAG by introducing a Retrieval Evaluator that classifies documents as Correct, Ambiguous, or Incorrect, and uses a scoring mechanism (-1 to 1) to evaluate relevance. It implements a decompose-then-recompose strategy for handling irrelevant retrievals and optionally triggers web searches when local retrieval fails. This approach identifies and handles low-quality or irrelevant retrievals, improving overall RAG pipeline robustness.

- **Model**: T5-large evaluator for document relevance scoring
- **Dataset**: [CRAG training data](https://github.com/HuskyInSalt/CRAG)
- **Training Approach**: SFT - reproduces original paper's training methodology exactly
- **Key Innovation**: Lightweight retrieval evaluator for efficient document scoring and filtering

For configuration and training setup, see the [Corrective RAG documentation](src/rag_model_training/corrective_rag/README.md).

## RQ-RAG

**Paper**: [Learning to Refine Queries for Retrieval Augmented Generation](https://arxiv.org/abs/2404.00610)

RQ-RAG improves retrieval by refining the query itself rather than the documents or output. Special control tokens (`[S_Rewritten_Query]`, `[S_Decomposed_Query]`, `[S_Disambiguated_Query]`) guide dynamic refinement through iterative cycles: Query Rewriting (improves original query based on context), Query Decomposition (breaks complex queries into focused sub-queries), and Query Disambiguation (resolves ambiguous queries). The model follows a tree-based decoding strategy for iterative refinement: generate → retrieve → refine → repeat.

- **Model**: Llama-3.2-3B with special token vocabulary expansion
- **Dataset**: [RQ-RAG training dataset](https://huggingface.co/datasets/zorowin123/rq_rag) (combines ARC-Easy/Challenge, OpenbookQA, HotpotQA, Musique, ASQA, LIMA, WizardLM, Open-Orca, OpenAssistant, GPT4-Alpaca)
- **Training Approach**: SFT - adapts original implementation to work with Llama-3.2 model family
- **Key Innovation**: Tree-based decoding with multiple refinement paths for complex queries

For configuration and training setup, see the [RQ-RAG documentation](src/rag_model_training/rq_rag/README.md).

## Self-RAG

**Paper**: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

Self-RAG introduces a framework for self-reflection during RAG using special reflection tokens that enable the model to evaluate its own decisions: Retrieval Decision (`[Retrieval]`/`[No Retrieval]`) determines whether to retrieve documents, Relevance (`[Relevant]`/`[Irrelevant]`) evaluates retrieved document quality, Grounding (`[Grounded]`/`[Not Grounded]`) verifies if generation is supported by evidence, and Utility (`[Utility:1]` through `[Utility:5]`) rates overall quality. The approach trains both a critic and generator model in two sequential phases: the critic learns to evaluate all four dimensions, and the generator learns to generate content and reflection tokens. This exemplifies training the model to evaluate both retrieval and generation decisions, applied here to a financial QA domain using the Earnings Calls dataset.

- **Models**: T5-base critic and generator models
- **Dataset**: [Earnings Calls Dataset](https://huggingface.co/datasets/lamini/earnings-calls-qa) (for financial QA domain)
- **Training Approach**: Two-phase SFT (critic phase → generator phase) for joint retrieval and generation reflection
- **Key Innovation**: Two-phase architecture enabling independent critic and generator training with modular components

For configuration and training setup, see the [Self-RAG documentation](src/rag_model_training/self_rag/README.md).

## Agentic RAG  

**GitHub:** [AutoDidact](https://github.com/dCaples/AutoDidact)

Agentic RAG trains language models as autonomous problem-solving agents using reinforcement learning (Group Relative Policy Optimization / GRPO). The model learns to plan and invoke tools strategically, recognizing when additional information is needed and autonomously searching for it. It develops capabilities including Self-Recognition of Missing Information (identifies when knowledge is insufficient), Query Rewriting and Refinement (autonomously improves queries based on results), Tool Call Generation (generates search queries and tool invocations), and learns through LLM-as-a-Judge Reward Functions that drive learning with correctness and formatting signals.

- **Model**: Llama-3.1-8B with LoRA-based parameter-efficient fine-tuning
- **Dataset**: [TriviaQA dataset](https://github.com/jcwleo/open-domain-qa) with FAISS-based retrieval corpus
- **Training Approach**: GRPO (Group Relative Policy Optimization) - reinforcement learning with two reward functions (Correctness: LLM-as-a-Judge evaluation, Formatting: proper search query generation)
- **Key Innovation**: Autonomous search capability with efficient LoRA training and vLLM integration; supports up to 6 search-refinement cycles per question

For configuration and training setup, see the [Agentic RAG documentation](src/rag_model_training/agentic_rag/README.md).

## ReZero Agentic RAG

**Paper**: [ReZero: Enhancing LLM search ability by trying one-more-time](https://arxiv.org/abs/2504.1001)

ReZero extends Agentic RAG with explicit persistence in information seeking and multi-cycle refinement. The key innovation is explicitly rewarding retry behavior—learning to recognize when initial search results are insufficient and autonomously deciding to refine queries and search again. Advanced composite rewards teach sophisticated search behavior: Search Retry and Persistence (learns to retry when initial attempts are insufficient), Search Strategy Optimization (develops effective strategies through learned composite rewards), and Iterative Refinement (uses 'think', 'search', and 'answer' tags for transparent reasoning). This enables strategic decisions about when to retry versus provide answers.

- **Model**: Llama-3.2-8B with LoRA-based parameter-efficient fine-tuning (rank 64)
- **Dataset**: [TriviaQA dataset](https://github.com/jcwleo/open-domain-qa) with multilingual-e5-large embeddings and FAISS indexing
- **Training Approach**: GRPO (Group Relative Policy Optimization) with six composite reward functions (Correctness: LLM-as-a-Judge evaluation, Formatting: proper tag usage, Retry: optimal retry behavior, EM Chunk: information presence in results, Search Strategy: reasoning quality, Search Diversity: varied query formulation)
- **Key Innovation**: Explicit reward for search persistence without excessive redundancy; supports up to 32 search-refinement cycles for improved answer quality

For configuration and training setup, see the [ReZero Agentic RAG documentation](src/rag_model_training/rezero_agentic_rag/README.md).

## Key Distinctions

### Retrieval Strategy & Quality Optimization (SFT)

- **Adaptive-RAG**, **Corrective RAG**, and **RQ-RAG** focus on optimizing the retrieval process itself
- Adaptive-RAG trains a classifier to select the appropriate retrieval strategy based on query complexity
- Corrective RAG trains an evaluator to assess document relevance and quality
- RQ-RAG trains query refinement models to improve retrieval through rewriting, decomposition, and disambiguation
- All use supervised learning with fixed labels from published datasets

### Joint Retrieval & Generation Reflection (Two-Phase SFT)

- **Self-RAG** takes a different approach by training the model to evaluate its own decisions during both retrieval and generation
- Uses a two-phase training process: critic phase (learns to evaluate all four dimensions) followed by generator phase (learns to generate with reflection tokens)
- Applies the framework to the Earnings Calls dataset, demonstrating how this joint training architecture works across different domains

### Agentic Generation with Autonomous Retrieval (GRPO)

- **Agentic RAG** and **ReZero Agentic RAG** focus on generation as the primary component, training the model to autonomously decide when and what to retrieve
- Both use reinforcement learning (GRPO) to learn effective search strategies through interaction and reward signals rather than fixed labels
- Agentic RAG: Learns to recognize missing information and generate search queries autonomously
- ReZero Agentic RAG: Extends this with explicit reward for retry behavior and sophisticated multi-cycle search strategies

## Repository Structure

Each technique is organized in its own directory under `src/rag_model_training/` with:

- **README.md**: Detailed technique documentation
- **train_{technique}.py** or **train.py**: Training script (e.g., `train_adaptive_rag.py`, `train.py` for RL approaches)
- **config.yaml**: Training configuration (or **config.py** for ReZero Agentic RAG)
- **requirements.txt**: Python dependencies
- **prepare_data.py** (where applicable): Dataset preparation script

## Getting Started

For training any of these techniques:

1. Install dependencies for your chosen approach:

   ```bash
   uv sync --group adaptive_rag    # for Adaptive-RAG
   uv sync --group corrective_rag  # for Corrective RAG
   uv sync --group rq_rag          # for RQ-RAG
   uv sync --group self_rag        # for Self-RAG
   uv sync --group agentic_rag     # for Agentic RAG
   uv sync --group rezero_agentic_rag  # for ReZero Agentic RAG
   ```

2. Navigate to the technique directory: `cd src/rag_model_training/<technique_name>`

3. Review and adjust configuration in the YAML/Python config file

4. Run training:
   - Adaptive-RAG: `python train_adaptive_rag.py`
   - Corrective RAG: `python train_corrective_rag.py`
   - RQ-RAG: `python train_rq_rag.py`
   - Self-RAG (two-phase): `python train_critic.py` then `python train_generator.py`
   - Agentic RAG: `python prepare_data.py` then `python train.py`
   - ReZero Agentic RAG: `python prepare_data.py` then `python train.py`

For detailed instructions specific to each technique, see the individual README files in each technique directory.
