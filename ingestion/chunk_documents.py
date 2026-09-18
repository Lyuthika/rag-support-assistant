import argparse

from load_documents import load_documents


DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    chunked_documents = []

    for document in documents:
        chunks = chunk_text(
            document["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(chunks):
            chunked_documents.append({
                "title": document["title"],
                "category": document["category"],
                "source_url": document["source_url"],
                "file_path": document["file_path"],
                "chunk_index": index,
                "content": chunk,
            })

    return chunked_documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents for RAG ingestion")

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of words per chunk",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of overlapping words between chunks",
    )

    args = parser.parse_args()

    documents = load_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Chunk size: {args.chunk_size} words")
    print(f"Chunk overlap: {args.chunk_overlap} words")
    print(f"Total chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']}")
        print(f"Title: {chunk['title']}")
        print(f"Category: {chunk['category']}")
        print(f"Words: {len(chunk['content'].split())}")
        print(f"Content preview: {chunk['content'][:300]}...")
        print("-" * 60)