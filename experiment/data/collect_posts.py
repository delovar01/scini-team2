import re
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# Каналы и условная основная тематика.
# topic потом при необходимости можно поправить вручную.
CHANNELS = {
    "hiaimedia": "ai",
    "llm_under_hood": "llm",
    "habr_ai": "ml_ai",
    "quant_prune_distill": "ml",
    "seeallochnaya": "ai_nlp",
}

POSTS_PER_CHANNEL = 50

OUTPUT_FILE = Path(__file__).parent / "posts.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    )
}


def parse_page(channel: str, before: int | None = None):
    url = f"https://t.me/s/{channel}"

    if before is not None:
        url += f"?before={before}"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    posts = []

    for message in soup.select(".tgme_widget_message"):
        data_post = message.get("data-post")

        if not data_post:
            continue

        # Например:
        # seeallochnaya/3737
        try:
            _, telegram_id = data_post.rsplit("/", 1)
            telegram_id = int(telegram_id)
        except ValueError:
            continue

        text_element = message.select_one(
        ".tgme_widget_message_text"
    )

        if text_element is None:
            continue

        text = text_element.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            continue

        text_without_urls = re.sub(
            r"https?://\S+",
            "",
            text
        ).strip()

        if len(text_without_urls) < 100:
            continue

        time_element = message.select_one("time")

        if time_element is None:
            continue

        datetime_value = time_element.get("datetime")

        if not datetime_value:
            continue

        # Telegram обычно отдаёт ISO datetime:
        # 2026-10-01T10:20:30+00:00
        date = datetime_value[:10]

        posts.append(
            {
                "telegram_id": telegram_id,
                "date": date,
                "text": text,
            }
        )

    return posts


def collect_channel(
    channel: str,
    topic: str,
    limit: int,
):
    result = []
    seen_ids = set()

    before = None

    while len(result) < limit:
        print(
            f"[{channel}] собрано "
            f"{len(result)}/{limit}"
        )

        try:
            page_posts = parse_page(
                channel,
                before,
            )
        except requests.RequestException as error:
            print(
                f"[{channel}] ошибка запроса: {error}"
            )
            break

        if not page_posts:
            print(
                f"[{channel}] больше постов не найдено"
            )
            break

        # Telegram возвращает сообщения не всегда
        # в удобном нам порядке.
        page_posts.sort(
            key=lambda post: post["telegram_id"],
            reverse=True,
        )

        for post in page_posts:
            telegram_id = post["telegram_id"]

            if telegram_id in seen_ids:
                continue

            seen_ids.add(telegram_id)

            index = len(result) + 1

            result.append(
                {
                    "id": f"{channel}_{index:04d}",
                    "channel": channel,
                    "date": post["date"],
                    "topic": topic,
                    "label": "clean",
                    "synthetic": False,
                    "text": post["text"],
                }
            )

            if len(result) >= limit:
                break

        # Идём дальше назад по истории канала
        smallest_id = min(
            post["telegram_id"]
            for post in page_posts
        )

        if before == smallest_id:
            break

        before = smallest_id

        # Не долбим Telegram запросами
        time.sleep(1)

    return result


def main():
    all_posts = []

    for channel, topic in CHANNELS.items():
        print()
        print(f"=== {channel} ===")

        posts = collect_channel(
            channel=channel,
            topic=topic,
            limit=POSTS_PER_CHANNEL,
        )

        all_posts.extend(posts)

        print(
            f"[{channel}] итог: {len(posts)}"
        )

        time.sleep(1)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        for post in all_posts:
            file.write(
                json.dumps(
                    post,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Готово: {len(all_posts)} постов"
    )
    print(
        f"Файл: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()