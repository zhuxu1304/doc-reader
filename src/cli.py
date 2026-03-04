from rich import print

from retriever import DenseRetriever, load_corpus
from qa import answer


def main():
    items = load_corpus()
    r = DenseRetriever()
    r.index(items)

    print("[bold cyan]Mini RAG CLI ready. Type your question, or 'exit'.[/bold cyan]")

    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        top = r.topk(q, k=3)
        print("\n[bold yellow]Retrieved evidence:[/bold yellow]")
        for item, score in top:
            print(f"- {item['id']} score={score:.3f} :: {item['text']}")

        out = answer(q, top)
        print("\n[bold green]Answer:[/bold green]")
        print(out.model_dump())


if __name__ == "__main__":
    main()