from src.pipeline.interview_pipeline import InterviewPipeline


def run_interview():
    """
    Run an interactive AI interview in the terminal.
    """

    print("\n" + "=" * 80)
    print("AI INTERVIEWER")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Get interview setup
    # --------------------------------------------------

    role = input("\nEnter the job role: ").strip()

    print("\nPaste the job description.")
    print("Press Enter twice when finished.\n")

    job_description_lines = []

    while True:
        line = input()

        if line == "":
            break

        job_description_lines.append(line)

    job_description = "\n".join(job_description_lines)

    if not role:
        print("A job role is required.")
        return

    if not job_description:
        print("A job description is required.")
        return

    # --------------------------------------------------
    # 2. Initialize RAG + memory pipeline
    # --------------------------------------------------

    print("\nPreparing interview...\n")

    pipeline = InterviewPipeline()

    # --------------------------------------------------
    # 3. Start with an opening question
    # --------------------------------------------------

    current_question = (
        f"Tell me about yourself and your experience "
        f"relevant to the {role} position."
    )

    turn_number = 1

    print("=" * 80)
    print("INTERVIEW STARTED")
    print("=" * 80)

    # --------------------------------------------------
    # 4. Interview loop
    # --------------------------------------------------

    while True:

        print(f"\nTURN {turn_number}")

        print("\nINTERVIEWER:")
        print(current_question)

        print("\nYOUR ANSWER:")
        candidate_answer = input("> ").strip()

        # Allow user to exit interview
        if candidate_answer.lower() in {
            "quit",
            "exit",
            "stop",
        }:
            break

        if not candidate_answer:
            print("\nPlease provide an answer.")
            continue

        print("\nGenerating next question...")

        try:
            result = pipeline.generate_next_question(
                role=role,
                job_description=job_description,
                current_question=current_question,
                candidate_answer=candidate_answer,
                top_k=5,
            )

        except Exception as error:
            print("\nAn error occurred:")
            print(error)
            break

        current_question = result["next_question"]

        turn_number += 1

    # --------------------------------------------------
    # 5. End interview
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("INTERVIEW ENDED")
    print("=" * 80)

    print(f"\nCompleted turns: {len(pipeline.memory)}")

    if len(pipeline.memory) > 0:

        print("\nINTERVIEW HISTORY")
        print("-" * 80)

        print(
            pipeline.memory.format_history(
                max_turns=100
            )
        )


def main():
    run_interview()


if __name__ == "__main__":
    main()
