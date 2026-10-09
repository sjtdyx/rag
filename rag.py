### COMPLETE THE CODE ###

from policy_loader import load_policy_documents
from rag_tool import retrieve_documents
from agent import load_model, generate_answer


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


def run_rag(question, model, tokenizer):

    retrieved = retrieve_documents(question)

    if not retrieved:
        context = "No relevant policy information was found."
    else:
        context = "\n\n".join(
            doc.page_content
            for doc in retrieved
        )

    answer = generate_answer(
        question,
        context,
        model,
        tokenizer
    )

    return answer


def main():

    model, tokenizer = load_model()

    question = (
        "What exact phone number should I call "
        "for the University IT Service Desk?"
    )

    print("QUESTION:")
    print(question)

    answer = run_rag(
        question,
        model,
        tokenizer
    )

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()