from query_data import query_rag
from mistralai import Mistral
from query_data import MISTRAL_API_KEY

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def run_interactive_rag():
    print("Interactive RAG (type 'exit' to quit)\n")

    while True:
        question = input("Enter your question: ").strip()
        if not question or question.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye! 👋")
            break

        # Run the RAG system
        print("\nRunning RAG for your query…………………………………!")
        response_text = query_rag(question)
        print(f"RAG Answer:\n{response_text}\n")

        # Ask if the user wants to evaluate
        do_eval = input("Do you want to evaluate this answer? (y/n): ").strip().lower()
        if do_eval in ["y", "yes"]:
            expected = input("Enter expected answer: ").strip()
            if expected:
                evaluate_answer(response_text, expected)
            else:
                print("No expected answer entered — skipping evaluation.!!!!!!!!!\n")
        print("-" * 50 + "\n")

def evaluate_answer(actual_response: str, expected_response: str):
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=actual_response,
    )

    # Call Mistral API for evaluation
    client = Mistral(api_key=MISTRAL_API_KEY)
    eval_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )

    evaluation = eval_response.choices[0].message.content.strip().lower()
    print(f"\nEvaluation Prompt:\n{prompt}\n")
    print("Mistral Evaluation Result: ", evaluation)

    if "true" in evaluation:
        print("\033[92m✔️ The answer matches expected response.\033[0m\n")
    elif "false" in evaluation:
        print("\033[91m❌ The answer does NOT match expected response.\033[0m\n")
    else:
        print("⚠️ Could not clearly determine true/false from evaluation.\n")

if __name__ == "__main__":
    run_interactive_rag()