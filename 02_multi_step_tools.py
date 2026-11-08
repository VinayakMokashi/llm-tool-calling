"""
Multi-step tool calling (LangChain + Groq).

A calculator agent with six arithmetic tools. Questions such as "the square
root of the product of the sum of 2 and 3 with 12.5" need the output of one
tool as the input of the next, so one round of tool calls is not enough.
The agent loops instead:

    ask the model -> run every tool it requests -> send the results back -> repeat

until the model answers without asking for a tool, or max_turns is reached.
This is the ReAct pattern (reason, act, observe) built on the model's native
tool calling. A failed tool call is sent back as an error message, so the model
can correct itself instead of the program crashing.

How to run
----------
1. Copy `.env.example` to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 02_multi_step_tools.py

With no arguments it solves two sample problems. Use --question to ask your own.
"""

import argparse
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "openai/gpt-oss-120b"

SAMPLE_QUESTIONS = [
    "What is the square root of the product of the sum of 2 and 3 with 12.5?",
    "Divide 1000 by 8, subtract 25, then raise the result to the power of 2.",
]

SYSTEM = SystemMessage(
    content=(
        "You are a careful calculator. Use the provided tools for every arithmetic step; "
        "never do the math yourself. Work one step at a time and feed the result of one tool "
        "into the next. When steps do not depend on each other, request them in the same turn. "
        "When you have the final number, state it plainly."
    )
)


@tool
def add(numbers: list[float]) -> float:
    """Add a list of numbers together and return the sum."""
    return sum(numbers)


@tool
def multiply(numbers: list[float]) -> float:
    """Multiply a list of numbers together and return the product."""
    product = 1.0
    for n in numbers:
        product *= n
    return product


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def square_root(number: float) -> float:
    """Calculate the square root of a non-negative number."""
    if number < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(number)


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the given exponent."""
    return base ** exponent


TOOLS = [add, multiply, subtract, divide, square_root, power]
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


def chat_with_tools(llm, question, max_turns=20, verbose=True):
    """Run the model/tool conversation until no more tool calls are requested."""
    messages = [SYSTEM, HumanMessage(content=question)]

    for turn in range(1, max_turns + 1):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return answer_text(response)

        for call in response.tool_calls:
            try:
                result = TOOL_MAP[call["name"]].invoke(call["args"])
            except Exception as exc:  # send the error back so the model can recover
                result = f"Error: {exc}"
            if verbose:
                print(f"  [turn {turn}] {call['name']}({call['args']}) -> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: hit the max_turns limit before a final answer."


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Solve multi-step arithmetic with a tool-calling loop.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    parser.add_argument("--max-turns", type=int, default=20, help="maximum rounds of tool calls")
    args = parser.parse_args(argv)

    llm = make_llm()
    for question in [args.question] if args.question else SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {chat_with_tools(llm, question, args.max_turns)}\n")


if __name__ == "__main__":
    main()
