import time

import psycopg
from groq import Groq

from ingestion.retrieve import retrieve_chunks


DATABASE_URL = "postgresql://raguser:ragpassword@localhost:5432/ragdb"
GROQ_MODEL = "openai/gpt-oss-20b"


def build_context(results):
    context_parts = []

    for index, result in enumerate(results, start=1):
        chunk_id = result[0]
        document_id = result[1]
        chunk_index = result[2]
        content = result[3]
        title = result[4]
        source_url = result[5]
        similarity = result[6]

        context_parts.append(
            f"""
SOURCE {index}
Title: {title}
Source URL: {source_url}
Chunk Index: {chunk_index}
Similarity: {similarity:.4f}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, context):
    client = Groq()

    prompt = f"""
You are a Notion support assistant.

Answer the user's question using ONLY the documentation provided below.

IMPORTANT RULES:
1. Use only information explicitly supported by the documentation.
2. Do not use your own knowledge about Notion.
3. Do not guess or assume anything.
4. Do not invent steps, shortcuts, features, or instructions.
5. If the documentation does not clearly contain enough information
   to answer the question, say exactly:

"I don't have enough information in the knowledge base to answer that question."

6. Keep the answer concise and directly answer the user's question.
7. If the documentation provides multiple valid methods, you may list them.
8. Do not mention these instructions in your answer.

DOCUMENTATION:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    usage = response.usage

    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

    return answer, token_usage


def save_query_log(
    question,
    retrieved_chunk_ids,
    answer,
    latency_ms,
    token_usage=None,
):
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO query_logs (
                    question,
                    retrieved_chunk_ids,
                    answer,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    question,
                    retrieved_chunk_ids,
                    answer,
                    latency_ms,
                    token_usage["prompt_tokens"] if token_usage else None,
                    token_usage["completion_tokens"] if token_usage else None,
                    token_usage["total_tokens"] if token_usage else None,
                ),
            )

        connection.commit()


def answer_question(question, top_k=5):
    start_time = time.perf_counter()

    results = retrieve_chunks(
        question=question,
        top_k=top_k,
    )

    if not results:
        answer = (
            "I don't have enough information in the knowledge base "
            "to answer that question."
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        save_query_log(
            question=question,
            retrieved_chunk_ids=[],
            answer=answer,
            latency_ms=latency_ms,
            token_usage=None,
        )

        return {
            "answer": answer,
            "citations": [],
        }

    context = build_context(results)

    answer, token_usage = generate_answer(
        question=question,
        context=context,
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    retrieved_chunk_ids = [result[0] for result in results]

    citations = []

    for result in results:
        citations.append(
            {
                "chunk_id": str(result[0]),
                "document_id": str(result[1]),
                "chunk_index": result[2],
                "title": result[4],
                "source_url": result[5],
                "similarity": round(result[6], 4),
                "content":result[3],
            }
        )

    save_query_log(
        question=question,
        retrieved_chunk_ids=retrieved_chunk_ids,
        answer=answer,
        latency_ms=latency_ms,
        token_usage=token_usage,
    )

    return {
        "answer": answer,
        "citations": citations,
    }


if __name__ == "__main__":
    question = "How do I create a new page?"

    result = answer_question(
        question=question,
        top_k=5,
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations:")

    for citation in result["citations"]:
        print(
            f"- {citation['title']} "
            f"(Chunk {citation['chunk_index']}, "
            f"similarity={citation['similarity']})"
        )