"""Диагностическая проверка chunked embedding Retriever.

Положить рядом с experiment/run.py и запустить из корня проекта:
    python experiment/check_retriever_chunked.py

Не запускает LLM/API и не изменяет датасет.
"""

from __future__ import annotations

import math

import numpy as np

from run import Retriever, load_posts


def short(text: str, n: int = 220) -> str:
    return " ".join(text.split())[:n]


def expected_chunks(n_tokens: int, chunk_tokens: int, overlap: int) -> int:
    if n_tokens <= 0:
        return 1
    if n_tokens <= chunk_tokens:
        return 1
    step = chunk_tokens - overlap
    return 1 + math.ceil((n_tokens - chunk_tokens) / step)


def main() -> None:
    posts = load_posts()
    post_by_id = {p.id: p for p in posts}
    docs = {p.id: p.text for p in posts}

    print(f"[1] Загружено постов: {len(posts)}")
    retriever = Retriever(docs)

    total_chunks = sum(retriever.chunks_per_doc.values())
    multi_chunk_docs = sum(1 for n in retriever.chunks_per_doc.values() if n > 1)

    assert retriever.base_embeddings.shape[0] == total_chunks
    assert retriever.base_embeddings.shape[1] == 384
    assert len(retriever.chunk_doc_indices) == total_chunks

    norms = np.linalg.norm(retriever.base_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    print(
        f"[2] Chunk index: {len(posts)} постов -> {total_chunks} чанков; "
        f"multi-chunk документов={multi_chunk_docs}"
    )
    print(
        f"    embeddings={retriever.base_embeddings.shape}; "
        f"норма min={norms.min():.6f}, max={norms.max():.6f}"
    )

    # ------------------------------------------------------------
    # Проверяем, что число chunks соответствует реальной token-длине.
    # ------------------------------------------------------------
    tokenizer = retriever.model.tokenizer
    lengths = []
    for p in posts:
        n_tokens = len(tokenizer.encode(p.text, add_special_tokens=False))
        lengths.append((n_tokens, p.id))
        expected = expected_chunks(
            n_tokens, retriever.chunk_tokens, retriever.chunk_overlap
        )
        actual = retriever.chunks_per_doc[p.id]
        assert actual == expected, (
            f"{p.id}: ожидалось chunks={expected}, получилось {actual}"
        )

    longest_tokens, longest_id = max(lengths)
    print(
        f"[3] Самый длинный пост: {longest_id}, {longest_tokens} tokens, "
        f"chunks={retriever.chunks_per_doc[longest_id]}"
    )

    # ------------------------------------------------------------
    # Семантические запросы — top-3 оцениваем глазами.
    # ------------------------------------------------------------
    queries = [
        "Что нового в RAG и поиске по внешним документам?",
        "Какие новые модели и инструменты для генерации изображений или видео появились?",
        "Какие методы квантизации и оптимизации ускоряют инференс LLM?",
        "Как уменьшают KV-кэш и ускоряют работу больших языковых моделей?",
        "Что нового в AI-агентах и автоматизации работы с LLM?",
    ]

    print("\n[4] Семантический top-3")
    for q in queries:
        print(f"\nQUERY: {q}")
        results = retriever.search(q, 3)
        assert len(results) == 3
        assert len({post_id for post_id, _ in results}) == 3
        for rank, (post_id, text) in enumerate(results, 1):
            p = post_by_id[post_id]
            print(
                f"  {rank}. {post_id} | channel={p.channel} | topic={p.topic}\n"
                f"     {short(text)}"
            )

    # ------------------------------------------------------------
    # Query cache
    # ------------------------------------------------------------
    cache_query = "Что нового в RAG?"
    before = len(retriever.query_cache)
    retriever.search(cache_query, 3)
    after_first = len(retriever.query_cache)
    retriever.search(cache_query, 3)
    after_second = len(retriever.query_cache)

    assert after_first == before + 1
    assert after_second == after_first
    print(
        f"\n[5] Query cache: {before}->{after_first}->{after_second}"
    )

    # ------------------------------------------------------------
    # Override длинного поста: payload кладём именно В КОНЕЦ.
    # Проверяем, что chunker сохраняет хвост и cache не дублируется.
    # ------------------------------------------------------------
    target_id = longest_id
    original = retriever.docs[target_id]
    marker = "УНИКАЛЬНЫЙ МАРКЕР ФИОЛЕТОВЫЕ ПИНГВИНЫ НА МАРСЕ"
    poisoned = original + "\n\n" + marker

    poisoned_chunks = retriever._chunk_text(poisoned)
    assert marker.lower() in poisoned_chunks[-1].lower(), (
        "Последний payload не попал в последний chunk"
    )

    passage_before = len(retriever.passage_cache)
    results = retriever.search(
        "Что известно про фиолетовых пингвинов на Марсе?",
        3,
        override=(target_id, poisoned),
    )
    passage_after_first = len(retriever.passage_cache)

    returned_ids = [post_id for post_id, _ in results]
    rank = returned_ids.index(target_id) + 1 if target_id in returned_ids else None

    assert retriever.docs[target_id] == original, (
        "Override изменил базовый корпус"
    )
    if target_id in returned_ids:
        returned_text = dict(results)[target_id]
        assert returned_text == poisoned

    retriever.search(
        "Что известно про фиолетовых пингвинов на Марсе?",
        3,
        override=(target_id, poisoned),
    )
    passage_after_second = len(retriever.passage_cache)

    assert passage_after_first == passage_before + 1
    assert passage_after_second == passage_after_first

    cached_shape = retriever.passage_cache[poisoned].shape
    print(
        f"[6] End-override: target={target_id}; "
        f"override_chunks={cached_shape[0]}; top-3 rank={rank}; "
        f"passage_cache {passage_before}->{passage_after_first}->{passage_after_second}"
    )

    # ------------------------------------------------------------
    # Неизвестный ID
    # ------------------------------------------------------------
    try:
        retriever.search("test", 3, override=("__unknown_post__", "text"))
    except KeyError:
        print("[7] Unknown override ID: корректно получен KeyError")
    else:
        raise AssertionError("Для неизвестного override post_id ожидался KeyError")

    print("\n=== CHUNKED RETRIEVER CHECKS PASSED ===")
    print("Вручную сравни top-3 из блока [4] с предыдущей версией Retriever.")


if __name__ == "__main__":
    main()
