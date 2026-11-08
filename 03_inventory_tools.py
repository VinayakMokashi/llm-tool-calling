"""
Inventory assistant with four tools (LangChain + Groq).

A small warehouse is held in memory (five SKUs with stock, reorder points,
unit costs and daily demand). The model answers stock questions in plain
English by choosing among four tools:

- get_stock_level            - stock and reorder point for one SKU
- list_low_stock_items       - every SKU at or below its reorder point
- days_until_stockout        - days of stock left at the current demand rate
- calculate_reorder_quantity - how much to order for a given supplier lead time

The model picks the tool and fills in its arguments (for example
lead_time_days=10 from "my supplier takes 10 days"). The same loop as in
02_multi_step_tools.py lets one question use several rounds of tools, such as
listing the low-stock items and then sizing an order for each.

How to run
----------
1. Copy `.env.example` to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 03_inventory_tools.py

With no arguments it asks five sample questions. Use --question to ask your own.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "openai/gpt-oss-120b"

INVENTORY = {
    "SKU-001": {"name": "Industrial Valve", "stock": 120, "reorder_point": 50, "unit_cost": 84.50},
    "SKU-002": {"name": "Hydraulic Pump", "stock": 18, "reorder_point": 25, "unit_cost": 430.00},
    "SKU-003": {"name": "Conveyor Belt", "stock": 5, "reorder_point": 10, "unit_cost": 210.75},
    "SKU-004": {"name": "Safety Helmet", "stock": 300, "reorder_point": 100, "unit_cost": 12.00},
    "SKU-005": {"name": "Pressure Gauge", "stock": 42, "reorder_point": 40, "unit_cost": 55.00},
}

# Average units sold per day
DAILY_DEMAND = {"SKU-001": 8, "SKU-002": 3, "SKU-003": 2, "SKU-004": 15, "SKU-005": 6}

SAMPLE_QUESTIONS = [
    "What is the current stock level for SKU-003?",
    "Which items need to be reordered right now?",
    "How many days until SKU-002 runs out of stock?",
    "What quantity of SKU-003 should I order if my supplier takes 10 days to deliver?",
    # needs two rounds: find the low-stock items, then size an order for each
    "Which items need reordering, and how many units of each should I order with a 7-day lead time?",
]


@tool
def get_stock_level(sku: str) -> str:
    """Return current stock level and reorder point for a given SKU."""
    item = INVENTORY.get(sku.upper())
    if not item:
        return f"SKU {sku} not found."
    status = "LOW" if item["stock"] <= item["reorder_point"] else "OK"
    return (
        f"{item['name']} ({sku}): stock={item['stock']}, "
        f"reorder_point={item['reorder_point']}, status={status}"
    )


@tool
def calculate_reorder_quantity(sku: str, lead_time_days: int = 7) -> str:
    """Calculate recommended reorder quantity for a SKU given supplier lead time in days."""
    item = INVENTORY.get(sku.upper())
    if not item:
        return f"SKU {sku} not found."
    daily = DAILY_DEMAND.get(sku.upper(), 1)
    # Order up to two lead times of demand: one to cover the wait, one as safety stock.
    safety_stock = daily * lead_time_days
    reorder_qty = max(daily * lead_time_days * 2 - item["stock"], 0)
    total_cost = reorder_qty * item["unit_cost"]
    return (
        f"{item['name']} ({sku}): recommend ordering {reorder_qty} units "
        f"(safety_stock={safety_stock}, lead_time={lead_time_days}d, "
        f"estimated_cost=${total_cost:,.2f})"
    )


@tool
def days_until_stockout(sku: str) -> str:
    """Estimate how many days until a SKU runs out of stock at current demand rate."""
    item = INVENTORY.get(sku.upper())
    if not item:
        return f"SKU {sku} not found."
    daily = DAILY_DEMAND.get(sku.upper(), 1)
    return (
        f"{item['name']} ({sku}): {item['stock'] / daily:.1f} days until stockout "
        f"(stock={item['stock']}, daily_demand={daily})"
    )


@tool
def list_low_stock_items() -> str:
    """List all SKUs that are at or below their reorder point and need restocking."""
    low = [
        f"{sku} ({item['name']}): stock={item['stock']}, reorder_point={item['reorder_point']}"
        for sku, item in INVENTORY.items()
        if item["stock"] <= item["reorder_point"]
    ]
    if not low:
        return "All items are adequately stocked."
    return f"Items needing reorder ({len(low)} total): " + "; ".join(low)


TOOLS = [get_stock_level, calculate_reorder_quantity, days_until_stockout, list_low_stock_items]
TOOL_MAP = {t.name: t for t in TOOLS}


def make_llm():
    """Create the Groq chat model with the tools attached."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key.")
    return ChatGroq(model=MODEL, temperature=0, max_retries=5).bind_tools(TOOLS)


def answer_text(response):
    """The model's answer, flagged if gpt-oss's hidden reasoning used up the token limit."""
    if response.response_metadata.get("finish_reason") == "length":
        return "(Answer cut off by the token limit. Try a narrower question.)"
    return response.content


def chat_with_tools(llm, question, max_turns=10):
    """Loop, because one answer can need several rounds of tool calls."""
    messages = [HumanMessage(content=question)]

    for _ in range(max_turns):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return answer_text(response)

        for call in response.tool_calls:
            try:
                result = TOOL_MAP[call["name"]].invoke(call["args"])
            except Exception as exc:  # send the error back so the model can recover
                result = f"Error: {exc}"
            print(f"  tool: {call['name']}({call['args']})")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: hit the max_turns limit before a final answer."


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Answer inventory questions with four tools.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    args = parser.parse_args(argv)

    llm = make_llm()
    for question in [args.question] if args.question else SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {chat_with_tools(llm, question)}\n")


if __name__ == "__main__":
    main()
