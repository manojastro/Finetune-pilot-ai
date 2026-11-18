"""Generate the tiny synthetic tool-calling dataset in data/sample/.

Run:  uv run python scripts/make_sample_data.py

Output (committed to the repo):
  data/sample/train.jsonl  60 rows (10 per tool)
  data/sample/test.jsonl   20 rows (4 for two tools, 3 for the other four)

Each row stores raw chat `messages` and the `tools` list. The chat template is applied
later, at load time (see finetunepilot.data), so the dataset is not tied to one model.

To keep the test set honest, train and test use DIFFERENT sentence templates and
DIFFERENT argument values (cities, amounts, names...). The script fails if any request
text appears in both splits.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SEED = 1234


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _fn(
        "get_weather",
        "Get the current weather for a city.",
        {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        ["city"],
    ),
    _fn(
        "create_calendar_event",
        "Create an event in the user's calendar.",
        {
            "title": {"type": "string"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
            "time": {"type": "string", "description": "Start time in 24h HH:MM format"},
            "duration_minutes": {"type": "integer"},
        },
        ["title", "date"],
    ),
    _fn(
        "send_email",
        "Send an email.",
        {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        ["to", "subject", "body"],
    ),
    _fn(
        "calculate",
        "Evaluate an arithmetic expression.",
        {"expression": {"type": "string", "description": "e.g. 12 * (3 + 4)"}},
        ["expression"],
    ),
    _fn(
        "search_files",
        "Search the user's files.",
        {
            "query": {"type": "string"},
            "file_type": {"type": "string", "enum": ["pdf", "docx", "xlsx", "txt"]},
            "max_results": {"type": "integer"},
        },
        ["query"],
    ),
    _fn(
        "convert_currency",
        "Convert an amount from one currency to another.",
        {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "3-letter ISO code"},
            "to_currency": {"type": "string", "description": "3-letter ISO code"},
        },
        ["amount", "from_currency", "to_currency"],
    ),
]

# Each generator returns (request_text, arguments). `split` picks disjoint templates and values.
Generator = Callable[[random.Random, str], tuple[str, dict]]


def gen_weather(rng: random.Random, split: str) -> tuple[str, dict]:
    cities = {
        "train": ["Paris", "Tokyo", "Nairobi", "Lima", "Oslo", "Toronto", "Mumbai", "Cairo"],
        "test": ["Madrid", "Seoul", "Accra", "Quito"],
    }[split]
    city = rng.choice(cities)
    unit = rng.choice([None, "celsius", "fahrenheit"])
    if split == "train":
        templates = [
            "What's the weather like in {city}{u}?",
            "Is it raining in {city} right now{u}?",
            "Tell me the current weather for {city}{u}.",
        ]
    else:
        templates = ["How hot is it in {city} today{u}?", "Give me a weather update for {city}{u}."]
    u = f" in {unit}" if unit else ""
    args = {"city": city} | ({"unit": unit} if unit else {})
    return rng.choice(templates).format(city=city, u=u), args


def gen_calendar(rng: random.Random, split: str) -> tuple[str, dict]:
    titles = {
        "train": [
            "Team standup",
            "Dentist appointment",
            "Lunch with Priya",
            "Project review",
            "Gym session",
            "Call with the bank",
        ],
        "test": ["Parent-teacher meeting", "Budget planning", "Coffee with Arjun"],
    }[split]
    title = rng.choice(titles)
    month = rng.randint(1, 12) if split == "train" else rng.randint(1, 12)
    day = rng.randint(1, 14) if split == "train" else rng.randint(15, 28)
    date = f"2026-{month:02d}-{day:02d}"
    time = rng.choice(
        [None, "09:00", "13:30", "16:15"] if split == "train" else [None, "10:45", "18:00"]
    )
    duration = rng.choice([None, 30, 60] if split == "train" else [None, 45, 90])
    extra = ""
    if time:
        extra += f" at {time}"
    if duration:
        extra += f" for {duration} minutes"
    if split == "train":
        templates = [
            "Add '{title}' to my calendar on {date}{extra}.",
            "Schedule {title} on {date}{extra}.",
        ]
    else:
        templates = [
            "Put an event called '{title}' on {date}{extra}.",
            "Can you book '{title}' for {date}{extra}?",
        ]
    args = {"title": title, "date": date}
    if time:
        args["time"] = time
    if duration:
        args["duration_minutes"] = duration
    return rng.choice(templates).format(title=title, date=date, extra=extra), args


def gen_email(rng: random.Random, split: str) -> tuple[str, dict]:
    people = {
        "train": ["anna@example.com", "li.wei@example.org", "sam@example.net", "ravi@example.com"],
        "test": ["maria@example.org", "tom.k@example.com"],
    }[split]
    topics = {
        "train": [
            ("Meeting moved", "The meeting is moved to Friday."),
            ("Invoice", "Please find the invoice attached."),
            ("Thanks", "Thanks for your help today."),
            ("Running late", "I will be 10 minutes late."),
        ],
        "test": [
            ("Report ready", "The quarterly report is ready for review."),
            ("Holiday", "I am on holiday next week."),
        ],
    }[split]
    to = rng.choice(people)
    subject, body = rng.choice(topics)
    if split == "train":
        templates = [
            "Email {to} with subject '{subject}' saying: {body}",
            "Send an email to {to}. Subject: '{subject}'. Body: {body}",
        ]
    else:
        templates = [
            "Write to {to}, subject '{subject}', and tell them: {body}",
            "Please mail {to} about '{subject}' with the message: {body}",
        ]
    text = rng.choice(templates).format(to=to, subject=subject, body=body)
    return text, {"to": to, "subject": subject, "body": body}


def gen_calc(rng: random.Random, split: str) -> tuple[str, dict]:
    if split == "train":
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        templates = ["What is {e}?", "Calculate {e}.", "Compute {e} for me."]
    else:
        a, b = rng.randint(100, 999), rng.randint(2, 50)
        op = rng.choice(["*", "/"])
        templates = ["How much is {e}?", "Work out {e} please."]
    expr = f"{a} {op} {b}"
    return rng.choice(templates).format(e=expr), {"expression": expr}


def gen_search(rng: random.Random, split: str) -> tuple[str, dict]:
    queries = {
        "train": [
            "tax return 2025",
            "lease agreement",
            "holiday photos list",
            "meeting notes",
            "salary slip",
            "travel itinerary",
        ],
        "test": ["insurance policy", "project proposal", "grocery budget"],
    }[split]
    query = rng.choice(queries)
    ftype = rng.choice([None, "pdf", "docx", "xlsx", "txt"])
    max_results = rng.choice([None, 5, 10] if split == "train" else [None, 3, 20])
    t = f" {ftype}" if ftype else ""
    m = f" Show at most {max_results} results." if max_results else ""
    if split == "train":
        templates = ["Find my{t} files about '{q}'.{m}", "Search my{t} documents for '{q}'.{m}"]
    else:
        templates = ["Look through my{t} files for '{q}'.{m}", "Where is my{t} file on '{q}'?{m}"]
    args = {"query": query}
    if ftype:
        args["file_type"] = ftype
    if max_results:
        args["max_results"] = max_results
    return rng.choice(templates).format(t=t, q=query, m=m), args


def gen_currency(rng: random.Random, split: str) -> tuple[str, dict]:
    pairs = {
        "train": [("USD", "EUR"), ("GBP", "INR"), ("JPY", "USD"), ("EUR", "CHF"), ("AUD", "CAD")],
        "test": [("INR", "GBP"), ("CAD", "JPY"), ("CHF", "USD")],
    }[split]
    src, dst = rng.choice(pairs)
    amount = (
        rng.choice([50, 120, 250, 1000, 75.5])
        if split == "train"
        else rng.choice([300, 42.25, 5000])
    )
    if split == "train":
        templates = ["Convert {a} {s} to {d}.", "How much is {a} {s} in {d}?"]
    else:
        templates = ["I have {a} {s}. What is that in {d}?", "Change {a} {s} into {d}."]
    text = rng.choice(templates).format(a=amount, s=src, d=dst)
    return text, {"amount": amount, "from_currency": src, "to_currency": dst}


GENERATORS: list[tuple[str, Generator]] = [
    ("get_weather", gen_weather),
    ("create_calendar_event", gen_calendar),
    ("send_email", gen_email),
    ("calculate", gen_calc),
    ("search_files", gen_search),
    ("convert_currency", gen_currency),
]


def make_rows(split: str, per_tool: list[int], rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for (tool_name, gen), count in zip(GENERATORS, per_tool, strict=True):
        seen: set[str] = set()
        attempts = 0
        while len(seen) < count:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(f"Could not make {count} unique {split} rows for {tool_name}")
            text, args = gen(rng, split)
            if text in seen:
                continue
            seen.add(text)
            tools = list(TOOLS)
            rng.shuffle(tools)  # vary tool order so the model can't rely on position
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": args},
                                }
                            ],
                        },
                    ],
                    "tools": tools,
                }
            )
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"{split}-{i:03d}"
    return [{"id": r["id"], "messages": r["messages"], "tools": r["tools"]} for r in rows]


def main() -> None:
    rng = random.Random(SEED)
    train = make_rows("train", [10] * 6, rng)
    test = make_rows("test", [4, 4, 3, 3, 3, 3], rng)

    overlap = {r["messages"][0]["content"] for r in train} & {
        r["messages"][0]["content"] for r in test
    }
    if overlap:
        raise RuntimeError(f"Train/test overlap: {sorted(overlap)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
