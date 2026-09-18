import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET_PATH = Path("evaluation/golden_dataset.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_PATH = RESULTS_DIR / "retrieval_run.json"

API_URL = "http://127.0.0.1:8000/query"

TOP_K = 5
RECALL_KS = [1, 3, 5]

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
            f"API returned HTTP {error.code}: {error.read().decode('utf-8')}"
        )

    except URLError as error:
        raise RuntimeError(
            "Could not connect to the FastAPI server. "
            "Make sure uvicorn is running on http://127.0.0.1:8000"
        ) from error


def chunk_key(source_url, chunk_index):
    return (
        source_url,
        int(chunk_index),
    )


def get_expected_chunk_keys(question_data):
    return {
        chunk_key(
            item["source_url"],
            item["chunk_index"],
        )
        for item in question_data["expected_source_chunks"]
    }


def get_retrieved_chunk_keys(citations):
    return [
        chunk_key(
            citation["source_url"],
            citation["chunk_index"],
        )
        for citation in citations
    ]


def calculate_recall_at_k(expected_keys, retrieved_keys, k):
    if not expected_keys:
        return None

    retrieved_at_k = set(retrieved_keys[:k])

    hits = sum(
        1
        for expected_key in expected_keys
        if expected_key in retrieved_at_k
    )

    return hits / len(expected_keys)


def calculate_mrr(expected_keys, retrieved_keys):
    if not expected_keys:
        return None

    for rank, retrieved_key in enumerate(retrieved_keys, start=1):
        if retrieved_key in expected_keys:
            return 1 / rank

    return 0.0


def evaluate_question(question_data):
    question = question_data["question"]
    answerable = question_data["answerable"]

    response = call_query_api(question)

    answer = response["answer"]
    citations = response["citations"]

    retrieved_keys = get_retrieved_chunk_keys(citations)

    result = {
        "question": question,
        "answerable": answerable,
        "generated_answer": answer,
        "retrieved_chunks": [
            {
                "source_url": citation["source_url"],
                "chunk_index": citation["chunk_index"],
                "similarity": citation["similarity"],
            }
            for citation in citations
        ],
    }

    if answerable:
        expected_keys = get_expected_chunk_keys(question_data)

        result["recall_at_k"] = {}

        for k in RECALL_KS:
            result["recall_at_k"][f"recall@{k}"] = (
                calculate_recall_at_k(
                    expected_keys,
                    retrieved_keys,
                    k,
                )
            )

        result["mrr"] = calculate_mrr(
            expected_keys,
            retrieved_keys,
        )

    else:
        result["fallback_correct"] = (
            answer.strip() == FALLBACK_MESSAGE
            and len(citations) == 0
        )

    return result


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

    for k in RECALL_KS:
        values = [
            result["recall_at_k"][f"recall@{k}"]
            for result in answerable_results
        ]

        if values:
            summary[f"average_recall@{k}"] = sum(values) / len(values)
        else:
            summary[f"average_recall@{k}"] = None

    mrr_values = [
        result["mrr"]
        for result in answerable_results
    ]

    if mrr_values:
        summary["mean_mrr"] = sum(mrr_values) / len(mrr_values)
    else:
        summary["mean_mrr"] = None

    fallback_values = [
        result["fallback_correct"]
        for result in unanswerable_results
    ]

    if fallback_values:
        summary["unanswerable_fallback_accuracy"] = (
            sum(fallback_values) / len(fallback_values)
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
            "top_k": TOP_K,
            "recall_k_values": RECALL_KS,
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

    print("Starting retrieval evaluation...")
    print(f"Evaluation questions: {len(dataset)}")
    print(f"API: {API_URL}")
    print(f"Retriever top-k: {TOP_K}")
    print()

    results = []

    for index, question_data in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"Evaluating question {index}/{len(dataset)}..."
        )

        result = evaluate_question(question_data)

        results.append(result)

        print(f"Question: {result['question']}")

        if result["answerable"]:
            print(
                f"Recall@1: "
                f"{result['recall_at_k']['recall@1']:.4f}"
            )

            print(
                f"Recall@3: "
                f"{result['recall_at_k']['recall@3']:.4f}"
            )

            print(
                f"Recall@5: "
                f"{result['recall_at_k']['recall@5']:.4f}"
            )

            print(
                f"MRR: "
                f"{result['mrr']:.4f}"
            )

        else:
            print(
                f"Fallback correct: "
                f"{result['fallback_correct']}"
            )

        print("-" * 60)

    summary = calculate_summary(results)

    save_results(
        results,
        summary,
    )

    print()
    print("=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
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
        f"Average Recall@1: "
        f"{summary['average_recall@1']:.4f}"
    )

    print(
        f"Average Recall@3: "
        f"{summary['average_recall@3']:.4f}"
    )

    print(
        f"Average Recall@5: "
        f"{summary['average_recall@5']:.4f}"
    )

    print(
        f"Mean MRR: "
        f"{summary['mean_mrr']:.4f}"
    )

    print(
        f"Unanswerable fallback accuracy: "
        f"{summary['unanswerable_fallback_accuracy']:.4f}"
    )

    print()
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()