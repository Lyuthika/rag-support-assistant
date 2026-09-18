import psycopg
from sentence_transformers import SentenceTransformer

from chunk_documents import chunk_documents
from load_documents import load_documents


DATABASE_URL = "postgresql://raguser:ragpassword@localhost:5432/ragdb"
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def get_or_create_document(connection, document):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id
            FROM documents
            WHERE source_url = %s;
            """,
            (document["source_url"],),
        )

        existing = cursor.fetchone()

        if existing:
            document_id = existing[0]

            cursor.execute(
                """
                UPDATE documents
                SET title = %s,
                    category = %s,
                    content = %s
                WHERE id = %s;
                """,
                (
                    document["title"],
                    document["category"],
                    document["content"],
                    document_id,
                ),
            )

            print(f"Using existing document: {document['title']}")

        else:
            cursor.execute(
                """
                INSERT INTO documents (title, category, source_url, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    document["title"],
                    document["category"],
                    document["source_url"],
                    document["content"],
                ),
            )

            document_id = cursor.fetchone()[0]

            print(f"Created document: {document['title']}")

    return document_id


def insert_chunks(connection, document_id, chunks):
    with connection.cursor() as cursor:

        # Remove previous chunks so ingestion is safe to rerun.
        cursor.execute(
            """
            DELETE FROM chunks
            WHERE document_id = %s;
            """,
            (document_id,),
        )

        for chunk in chunks:
            embedding_text = str(chunk["embedding"])

            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s::vector);
                """,
                (
                    document_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    embedding_text,
                ),
            )


if __name__ == "__main__":
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    documents = load_documents()

    print(f"Documents found: {len(documents)}")

    with psycopg.connect(DATABASE_URL) as connection:

        for document in documents:

            document_id = get_or_create_document(
                connection,
                document,
            )

            chunks = chunk_documents(
                [document],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            texts = [chunk["content"] for chunk in chunks]

            embeddings = model.encode(
                texts,
                show_progress_bar=True,
            )

            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding.tolist()

            insert_chunks(
                connection,
                document_id,
                chunks,
            )

            print(f"Inserted {len(chunks)} chunks")

        connection.commit()

    print("\nIngestion complete.")