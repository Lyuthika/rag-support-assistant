import psycopg
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://raguser:ragpassword@localhost:5432/ragdb"
MODEL_NAME = "all-MiniLM-L6-v2"


SIMILARITY_THRESHOLD = 0.30
model = SentenceTransformer(MODEL_NAME)


def retrieve_chunks(question, top_k=3):

    question_embedding = model.encode(question)
    embedding_text = str(question_embedding.tolist())

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunks.id,
                    chunks.document_id,
                    chunks.chunk_index,
                    chunks.content,
                    documents.title,
                    documents.source_url,
                    1 - (chunks.embedding <=> %s::vector) AS similarity
                FROM chunks
                JOIN documents
                    ON chunks.document_id = documents.id
                ORDER BY chunks.embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding_text, embedding_text, top_k),
            )

            results = cursor.fetchall()

    filtered_results = []

    for result in results:
        similarity = result[6]

        if similarity >= SIMILARITY_THRESHOLD:
            filtered_results.append(result)

    return filtered_results


if __name__ == "__main__":
    question = "How do I create a new page?"

    results = retrieve_chunks(question, top_k=5)

    print(f"Question: {question}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Retrieved chunks: {len(results)}")
    print()

    if not results:
        print("No relevant information found.")
        print("The system should not generate an answer from the knowledge base.")

    else:
        for result in results:
            (
                chunk_id,
                document_id,
                chunk_index,
                content,
                title,
                source_url,
                similarity,
            ) = result

            print(f"Chunk index: {chunk_index}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Title: {title}")
            print(f"Source: {source_url}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Content: {content[:300]}...")
            print("-" * 70)