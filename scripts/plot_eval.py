import json
import os

import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@db:5432/app")


def load_eval_runs():
    engine = create_engine(DATABASE_URL, future=True)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT suite_name, metrics_json FROM eval_runs")).all()
    return [(row[0], row[1] if isinstance(row[1], dict) else json.loads(row[1])) for row in rows]


def plot_prompt_tokens(data):
    suites = [name for name, _ in data]
    tokens = [metrics.get("prompt_tokens_per_turn", 0) for _, metrics in data]
    plt.figure(figsize=(8, 4))
    plt.bar(suites, tokens)
    plt.ylabel("Prompt tokens per turn")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("prompt_tokens_per_turn.png")


def plot_success_rates(data):
    suites = [name for name, _ in data]
    success = [metrics.get("success_rate", 0) for _, metrics in data]
    plt.figure(figsize=(8, 4))
    plt.bar(suites, success, color="green")
    plt.ylabel("Success rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("success_rate.png")


if __name__ == "__main__":
    runs = load_eval_runs()
    if not runs:
        print("No eval runs found.")
    else:
        plot_prompt_tokens(runs)
        plot_success_rates(runs)
        print("Wrote prompt_tokens_per_turn.png and success_rate.png")

