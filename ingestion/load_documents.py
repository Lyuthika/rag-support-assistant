from pathlib import Path


DATA_DIR = Path("data/raw")


def load_documents():
    documents = []

    for file_path in DATA_DIR.rglob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        lines = content.splitlines()

        title = ""
        source_url = ""
        category = ""

        for line in lines:
            line = line.strip()

            if line.startswith("# ") and not title:
                title = line[2:].strip()
            elif line.startswith("Source URL:"):
                source_url = line.replace("Source URL:", "").strip()
            elif line.startswith("Category:"):
                category = line.replace("Category:", "").strip()

        if not title:
            title = file_path.stem.replace("-", " ").title()

        content_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# ") and stripped[2:].strip() == title:
                continue

            if stripped.startswith("Source URL:"):
                continue

            if stripped.startswith("Category:"):
                continue

            if stripped == "---":
                continue

            content_lines.append(line)

        clean_content = "\n".join(content_lines).strip()

        documents.append({
            "title": title,
            "category": category,
            "source_url": source_url,
            "content": clean_content,
            "file_path": str(file_path),
        })

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s)\n")

    for document in documents:
        print(f"Title: {document['title']}")
        print(f"Category: {document['category']}")
        print(f"Source: {document['source_url']}")
        print(f"File: {document['file_path']}")
        print(f"Characters: {len(document['content'])}")
        print("-" * 50)