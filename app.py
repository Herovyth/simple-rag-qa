from datasets import load_dataset

from chunking import chunk_text
from retriever import Retriever
from reranker import Reranker
from llm import generate_answer
from ui import create_ui
from pdf_loader import extract_text_from_pdf
from pdf_cache import (
    ensure_cache_dir,
    load_metadata,
    save_metadata,
    save_text,
    load_text
)


print("Loading dataset...")
dataset = load_dataset("prasad3458/Harry_Potter_Books")

ensure_cache_dir()
meta = load_metadata()

documents = []

for idx, row in enumerate(dataset["train"][0]):
    # якщо текст уже збережений — НЕ парсимо PDF
    cached_text = load_text(idx)

    if cached_text:
        print(f"Loaded cached doc {idx}")
        documents.append(cached_text)
        continue

    print(f"Extracting PDF {idx}")
    pdf = row["pdf"]
    text = extract_text_from_pdf(pdf)

    if text.strip():
        save_text(idx, text)
        documents.append(text)

    continue

meta["num_docs"] = len(documents)
save_metadata(meta)

print(f"Loaded {len(documents)} documents")

print("Chunking documents...")
chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))

print(f"Total chunks: {len(chunks)}")

if not chunks:
    raise RuntimeError("No chunks created. Extraction failed.")

print("Initializing components...")
retriever = Retriever(chunks)
reranker = Reranker()


def rag_pipeline(query, search_mode, api_key):
    retrieved = retriever.retrieve(
        query,
        mode=search_mode,
        top_k=10
    )

    reranked = reranker.rerank(
        query,
        retrieved,
        top_k=5
    )

    answer = generate_answer(
        query,
        reranked,
        api_key=api_key
    )

    formatted_chunks = []
    for i, chunk in enumerate(reranked, 1):
        formatted_chunks.append(f"[CHUNK {i}]\n{chunk}")

    return answer, "\n\n---\n\n".join(formatted_chunks)


if __name__ == "__main__":
    demo = create_ui(rag_pipeline)
    demo.launch()

