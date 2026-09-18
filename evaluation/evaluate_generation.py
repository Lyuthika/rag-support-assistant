import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from groq import Groq


DATASET_PATH = Path("evaluation/golden_dataset.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_PATH = RESULTS_DIR / "generation_run.json"

API_URL = "http://127.0.0.1:8000/query"

GROQ_MODEL = "openai/gpt-oss-20b"

FALLBACK_MESSAGE = (
    "I don't have enough information in the knowledge base "
    "to answer that question."
)


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def call_query_api(question):
    payload = json.dumps(
        {
            "question": question
        }
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)

    except HTTPError as error:
        raise RuntimeError(
            f"API returned HTTP {error.code}: "
            f"{error.read().decode('utf-8')}"
        )

    except URLError as error:
        raise RuntimeError(
            "Could not connect to the FastAPI server. "
            "Make sure uvicorn is running on "
            "http://127.0.0.1:8000"
        ) from error


def build_retrieved_context(citations):
    context_parts = []

    for index, citation in enumerate(citations, start=1):
        context_parts.append(
            f"""
SOURCE {index}
Title: {citation["title"]}
Source URL: {citation["source_url"]}
Chunk Index: {citation["chunk_index"]}

Content:
{citation["content"]}
"""
        )

    return "\n".join(context_parts)


def judge_answer(
    question,
    expected_answer,
    generated_answer,
    retrieved_context,
):
    client = Groq()

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation (RAG)
support assistant.

Evaluate the generated answer using ONLY the retrieved documentation.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED DOCUMENTATION:
{retrieved_context}

Evaluate two things.

1. FAITHFULNESS

Check whether the factual claims in the generated answer are
supported by the retrieved documentation.

Give:
1 if the important factual claims are supported.
0 if the answer contains important unsupported or invented claims.

2. ANSWER CORRECTNESS

Check whether the generated answer correctly answers the question
and agrees with the expected answer.

Give:
1 if the answer is correct.
0 if the answer is incorrect or misses the required information.

Return ONLY valid JSON in exactly this format:

{{
  "faithfulness": 0 or 1,
  "answer_correctness": 0 or 1,
  "reason": "short explanation"
}}

Do not include markdown.
Do not include ```json.
Do not add any text outside the JSON.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "LLM judge did not return valid JSON:\n"
            f"{content}"
        ) from error

    faithfulness = result.get("faithfulness")
    answer_correctness = result.get("answer_correctness")

    if faithfulness not in [0, 1]:
        raise ValueError(
            f"Invalid faithfulness score: {faithfulness}"
        )

    if answer_correctness not in [0, 1]:
        raise ValueError(
            f"Invalid answer correctness score: "
            f"{answer_correctness}"
        )

    return result


def evaluate_answerable_question(question_data, response):
    question = question_data["question"]
    expected_answer = question_data["expected_answer"]

    generated_answer = response["answer"]
    citations = response["citations"]

    retrieved_context = build_retrieved_context(citations)

    judge_result = judge_answer(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context,
    )

    return {
        "question": question,
        "answerable": True,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "faithfulness": judge_result["faithfulness"],
        "answer_correctness": judge_result["answer_correctness"],
        "judge_reason": judge_result["reason"],
        "retrieved_chunks": [
            {
                "source_url": citation["source_url"],
                "chunk_index": citation["chunk_index"],
                "similarity": citation["similarity"],
            }
            for citation in citations
        ],
    }


def evaluate_unanswerable_question(question_data, response):
    question = question_data["question"]

    generated_answer = response["answer"]
    citations = response["citations"]

    fallback_correct = (
        generated_answer.strip() == FALLBACK_MESSAGE
        and len(citations) == 0
    )

    return {
        "question": question,
        "answerable": False,
        "expected_answer": None,
        "generated_answer": generated_answer,
        "faithfulness": None,
        "answer_correctness": None,
        "judge_reason": None,
        "fallback_correct": fallback_correct,
        "retrieved_chunks": [],
    }


def calculate_summary(results):
    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    summary = {
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
    }

    faithfulness_values = [
        result["faithfulness"]
        for result in answerable_results
    ]

    correctness_values = [
        result["answer_correctness"]
        for result in answerable_results
    ]

    if faithfulness_values:
        summary["average_faithfulness"] = (
            sum(faithfulness_values)
            / len(faithfulness_values)
        )
    else:
        summary["average_faithfulness"] = None

    if correctness_values:
        summary["average_answer_correctness"] = (
            sum(correctness_values)
            / len(correctness_values)
        )
    else:
        summary["average_answer_correctness"] = None

    fallback_values = [
        result["fallback_correct"]
        for result in unanswerable_results
    ]

    if fallback_values:
        summary["unanswerable_fallback_accuracy"] = (
            sum(fallback_values)
            / len(fallback_values)
        )
    else:
        summary["unanswerable_fallback_accuracy"] = None

    return summary


def save_results(results, summary):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "configuration": {
            "api_url": API_URL,
            "groq_model": GROQ_MODEL,
        },
        "results": results,
        "summary": summary,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )


def main():
    dataset = load_dataset()

    print("Starting generation evaluation...")
    print(f"Evaluation questions: {len(dataset)}")
    print(f"API: {API_URL}")
    print(f"LLM judge: {GROQ_MODEL}")
    print()

    results = []

    for index, question_data in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"Evaluating question "
            f"{index}/{len(dataset)}..."
        )

        question = question_data["question"]

        print(f"Question: {question}")

        response = call_query_api(question)

        if question_data["answerable"]:
            result = evaluate_answerable_question(
                question_data,
                response,
            )

            print(
                f"Faithfulness: "
                f"{result['faithfulness']}"
            )

            print(
                f"Answer correctness: "
                f"{result['answer_correctness']}"
            )

            print(
                f"Judge reason: "
                f"{result['judge_reason']}"
            )

        else:
            result = evaluate_unanswerable_question(
                question_data,
                response,
            )

            print(
                f"Fallback correct: "
                f"{result['fallback_correct']}"
            )

        results.append(result)

        print("-" * 60)

    summary = calculate_summary(results)

    save_results(
        results,
        summary,
    )

    print()
    print("=" * 60)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Answerable questions: "
        f"{summary['answerable_questions']}"
    )

    print(
        f"Unanswerable questions: "
        f"{summary['unanswerable_questions']}"
    )

    print(
        f"Average Faithfulness: "
        f"{summary['average_faithfulness']:.4f}"
    )

    print(
        f"Average Answer Correctness: "
        f"{summary['average_answer_correctness']:.4f}"
    )

    print(
        f"Unanswerable fallback accuracy: "
        f"{summary['unanswerable_fallback_accuracy']:.4f}"
    )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()