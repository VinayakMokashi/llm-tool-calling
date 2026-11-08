"""
Single tool, single round (LangChain + Groq).

The smallest useful tool-calling program. The model gets one tool,
square_root, and decides on its own whether a question needs it:

1. The question goes to the model together with the tool's description.
2. If the model asks for the tool, this script runs it and sends the result
   back as a tool message.
3. The model writes the final answer from that result.

A question that needs no arithmetic ("What is the capital of France?") is
answered directly, with no tool call.

How to run
----------
1. Copy `.env.example` to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 01_single_tool.py

With no arguments it asks three sample questions. Use --question to ask your own.
"""

import argparse
import math
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

SAMPLE_QUESTIONS = [
    "Take the square root of 4",
    "What is the capital of France?",  # needs no tool, so the model answers directly
    "Can you calculate the square root of 2025 for me?",
]


@tool
def square_root(number: float) -> float:
    """Calculate the square root of a non-negative number."""
    if number < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(number)


TOOLS = [square_root]
TOOL_MAP = {t.name: t for t in TOOLS}


def make_llm():
    """Create the Groq chat model with the tools attached."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key.")
    # bind_tools sends each tool's name, docstring and argument types with every request.
    # max_retries rides out free-tier rate limits.
    return ChatGroq(model=MODEL, temperature=0, max_retries=5).bind_tools(TOOLS)


def run_tool(call):
    """Run one tool call. An error is returned as text so the model can explain it."""
    try:
        return TOOL_MAP[call["name"]].invoke(call["args"])
    except Exception as exc:
        return f"Error: {exc}"


def answer_text(response):
    """The model's answer, flagged if gpt-oss's hidden reasoning used up the token limit."""
    if response.response_metadata.get("finish_reason") == "length":
        return "(Answer cut off by the token limit. Try a narrower question.)"
    return response.content


def chat_with_tools(llm, question):
    """Ask once, run any tools the model requests, then ask again for the answer."""
    messages = [HumanMessage(content=question)]
    response = llm.invoke(messages)
    if not response.tool_calls:
        return answer_text(response)

    messages.append(response)  # the model's tool request must precede the tool results
    for call in response.tool_calls:
        result = run_tool(call)
        print(f"  tool: {call['name']}({call['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = llm.invoke(messages)
    if final.tool_calls:
        # This script allows one round only; 02_multi_step_tools.py loops instead.
        return "(The model asked for another round of tool calls, which this script does not run.)"
    return answer_text(final)


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Answer questions with one tool and one round of tool calls.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    args = parser.parse_args(argv)

    llm = make_llm()
    for question in [args.question] if args.question else SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {chat_with_tools(llm, question)}\n")


if __name__ == "__main__":
    main()
