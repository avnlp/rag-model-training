import nest_asyncio
from rl_helpers import run_agent
from search_module import get_qa_dataset

nest_asyncio.apply()


def run_eval(generate_fn, verify_fn, tokenizer):
    """Run evaluation on the agent.

    Args:
        generate_fn: Function to generate model responses
        verify_fn: Function to verify answers
        tokenizer: Tokenizer for processing text

    Returns:
        list: Full chat states from evaluation
    """
    train_dataset, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    agentic_outputs = run_agent(generate_fn, tokenizer, questions)
    full_chat_states = agentic_outputs.full_chat_states
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])

    print("RESULTS:")
    print("percentage of correct answers:", sum(rewards) / len(rewards))
    print("=" * 30)

    return full_chat_states
