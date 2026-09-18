from sentence_transformers import SentenceTransformer

from chunk_documents import chunk_documents
from load_documents import load_documents


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks, model):
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


if __name__ == "__main__":
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    documents = load_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=400,
        chunk_overlap=50,
    )

    chunks = create_embeddings(chunks, model)

    print()
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimensions: {len(chunks[0]['embedding'])}")

    print()
    print("First chunk:")
    print(f"Title: {chunks[0]['title']}")
    print(f"Chunk index: {chunks[0]['chunk_index']}")
    print(f"Embedding first 5 values: {chunks[0]['embedding'][:5]}")