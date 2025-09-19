# This code is based on the implementation from: https://github.com/dCaples/AutoDidact/blob/main/rl_helpers.py.

from collections.abc import Callable

import nest_asyncio

from .prompts import remove_reasoning

nest_asyncio.apply()


def verify(student_answer: str, answer: str) -> bool:
    """Verify if the student's answer matches the correct answer.

    This function performs a case-insensitive comparison of the student's answer
    and the ground truth answer after stripping whitespace.

    Args:
        student_answer: The model's answer to be verified.
        answer: The ground truth answer for comparison.

    Returns:
        True if the student's answer is correct, False otherwise.
    """
    # Simple string matching for now
    return student_answer.strip().lower() == answer.strip().lower()


# Verification
async def check_correctness(question, student_answer, answer):
    """Calculate reward for a given student answer.

    This function logs the question, student answer, and correct answer to
    a file.
    It then checks if the student answer starts with "Error during" or is
    too short.
    If not, it removes reasoning from the student answer and verifies it
    against
    the correct answer.

    Args:
        question: The original question.
        student_answer: The model's answer.
        answer: The ground truth answer.

    Returns:
        Reward value (1 for correct, 0 for incorrect).
    """
    # log to "./reward_func.log"
    with open("reward_func.log", "a") as f:
        f.write("\n" + "==" * 40 + "\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Student Answer: {student_answer}\n")
        f.write(f"Answer: {answer}\n")
        if student_answer.startswith("Error during"):
            f.write("failed function call")
            return 0
        if len(student_answer) < 5:
            f.write("failed Too short answer\n")
            return 0
        else:
            f.write("last message didn't fail\n")
            student_answer_clean = remove_reasoning(student_answer)
            is_correct = verify(student_answer_clean, answer)
            f.write(f"Is Correct: {is_correct}, so reward is {int(is_correct)}\n")
            return 1 if is_correct else 0


def check_student_answers(
    questions: list[str],
    answers: list[str],
    student_answers: list[str],
    vllm_generate_func: Callable[[list[str]], list[str]],
    tokenizer,
    log_file: str = "qa_log.txt",
) -> list[bool]:
    """Evaluates a list of student answers against the true answers using a
    vLLM generate function.

    The function applies the chat template to each prompt before passing it
    to the generate function.
    It also appends the details of each QA pair and the verifier's response
    to a log file.

    Args:
        questions: A list of strings representing the questions.
        answers: A list of strings representing the correct answers.
        student_answers: A list of strings containing the student's answers.
        vllm_generate_func: A function that takes a list of
            chat-formatted prompt strings and returns a list of generated
            outputs.
        tokenizer: The tokenizer used to apply the chat template.
        log_file: Path to the file where the QA pairs and verification
        responses
            will be appended. Defaults to "qa_log.txt".

    Returns:
        A list of booleans indicating whether each student's answer is correct.

    """
    if not (len(questions) == len(answers) == len(student_answers)):
        msg = "The number of questions, answers, and student answers must be equal."
        raise ValueError(msg)

    prompts = []
    for question, answer, student_ans in zip(questions, answers, student_answers, strict=False):
        # Construct the plain text prompt for each QA pair.
        prompt_text = (
            "You are grading a student's answer. For the following question, "
            "compare the student's answer to the correct answer. Reply with 'Yes' "
            "if the student's answer is correct, or 'No' if it is completely "
            "incorrect.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Student Answer: {student_ans}\n"
        )
        # Apply the chat template to the prompt.
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted_prompt)

    # Get the model responses in batch (each response should ideally be
    # "Yes" or "No")
    responses = vllm_generate_func(prompts)
    responses_text = []
    for response in responses:
        # Handle different response formats
        if hasattr(response, "outputs"):
            try:
                responses_text.append(response.outputs[0].text)
            except (AttributeError, IndexError):
                # Fallback for simple string responses
                responses_text.append(str(response))
        else:
            responses_text.append(str(response))

    # Evaluate each response and mark as correct if "yes" appears in the
    # answer (case-insensitive)
    results = []
    for response in responses_text:
        results.append("yes" in response.lower())

    # Append the QA details and verifier's response to the specified log file
    with open(log_file, "a") as file:
        for question, answer, student_ans, verifier_response in zip(
            questions, answers, student_answers, responses_text, strict=False
        ):
            file.write("Question: " + question + "\n")
            file.write("Correct Answer: " + answer + "\n")
            file.write("Student Answer: " + student_ans + "\n")
            file.write("Verifier said: " + verifier_response + "\n")
            file.write("-" * 40 + "\n")

    return results


def build_reward_correctness_fn(generate_fn, tokenizer):
    """Build a reward function based on answer correctness.

    Args:
        generate_fn: A function that generates responses using vLLM.
        tokenizer: The tokenizer used to apply chat templates.

    Returns:
        A reward function that calculates rewards based on answer correctness.
    """

    def reward_correctness(prompts, completions, **reward_kwargs):
        """Calculate rewards based on the correctness of student answers.

        Args:
            prompts: List of prompts.
            completions: List of completions from the model.
            **reward_kwargs: Additional reward arguments, including "answer".

        Returns:
            List of boolean values indicating correctness of each answer.
        """
        teacher_answers = reward_kwargs["answer"]
        student_answers = [completion["messages"][-1]["content"] for completion in completions]

        correct = check_student_answers(
            prompts,
            teacher_answers,
            student_answers,
            vllm_generate_func=generate_fn,
            tokenizer=tokenizer,
        )
        return correct

    return reward_correctness


def reward_formatting(prompts, completions, **reward_kwargs):
    """Reward based on answer formatting.

    This function checks if the completions contain any error function calls.
    It returns a reward of 0.7 for completions without errors, and 0 for
    those with errors.

    Args:
        prompts: List of prompts.
        completions: List of completions from the model.
        **reward_kwargs: Additional reward arguments.

    Returns:
        List of rewards (0.7 for no errors, 0 for errors).
    """
    # make sure full chats doesn't have any error function calls
    has_error = [False] * len(completions)
    for i, chat in enumerate(completions):
        for message in chat["messages"]:
            if "Error during" in message["content"]:
                has_error[i] = True
                break
    return [0.7 if not e else 0 for e in has_error]
