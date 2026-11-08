"""
MCP client that hands an MCP server's tools to an LLM (FastMCP + Groq).

The model never talks to MCP. This script sits between the two:

1. Start server.py as a subprocess and connect to it over stdio.
2. Ask the server which tools it has (MCP `tools/list`).
3. Convert each tool to the OpenAI function-calling format. MCP describes a
   tool's inputs with JSON Schema, which is what that format expects too, so
   the schema is copied across unchanged.
4. Send the question and the tools to the model.
5. Run every tool the model requests through MCP (`tools/call`), send the
   results back, and repeat until the model answers.

The client only learns about the tools at run time, so it would work with any
other MCP server by changing SERVER_PATH, and the server works with any
MCP-compatible app. The model is called through Groq's OpenAI-compatible
endpoint with the official openai SDK, so another OpenAI-compatible provider
only needs a different base_url and MODEL.

How to run
----------
1. Copy `.env.example` to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 04_mcp/client.py

With no arguments it asks three sample questions. Use --question to ask your
own, or --list-tools to see what the server offers without calling the model.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from openai import AsyncOpenAI

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "openai/gpt-oss-20b"
SERVER_PATH = Path(__file__).with_name("server.py")

SAMPLE_QUESTIONS = [
    "What time is it in Bangalore?",
    "What's the weather like in Bangalore?",
    "What is the capital of Japan?",  # needs no tool, so the model answers directly
]


def make_llm():
    """Create an OpenAI SDK client pointed at Groq's OpenAI-compatible endpoint."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key.")
    return AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                       max_retries=5)


def to_openai_tools(mcp_tools):
    """Convert MCP tool descriptions to the OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema},
        }
        for t in mcp_tools
    ]


async def call_mcp_tool(mcp_client, call):
    """Run one tool call on the MCP server and return its text result."""
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"Error: the arguments were not valid JSON ({exc})"
    # raise_on_error=False returns a failed call as text, so the model can see what went wrong
    result = await mcp_client.call_tool(call.function.name, args, raise_on_error=False)
    return args, "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def chat_with_tools(llm, mcp_client, tools, question, max_turns=5):
    """Loop: ask the model, run the tools it requests through MCP, until it answers."""
    messages = [{"role": "user", "content": question}]

    for _ in range(max_turns):
        response = await llm.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                                     tool_choice="auto")
        choice = response.choices[0]
        if not choice.message.tool_calls:
            if choice.finish_reason == "length":  # gpt-oss's hidden reasoning used up the token limit
                return "(Answer cut off by the token limit. Try a narrower question.)"
            return choice.message.content

        # The model's tool request must precede the tool results in the conversation.
        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in choice.message.tool_calls
            ],
        })
        for call in choice.message.tool_calls:
            args, text = await call_mcp_tool(mcp_client, call)
            print(f"  tool via MCP: {call.function.name}({args}) -> {text}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    return "Stopped: hit the max_turns limit before a final answer."


async def run(questions, list_only=False):
    # Passing a .py path makes FastMCP start it with this Python interpreter and talk over stdio.
    async with Client(SERVER_PATH) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        print(f"Connected to {SERVER_PATH.name}; tools: {', '.join(t.name for t in mcp_tools)}\n")

        if list_only:
            for t in mcp_tools:
                print(f"{t.name}: {t.description}")
                print(f"  input schema: {json.dumps(t.input_schema)}\n")
            return

        llm = make_llm()
        tools = to_openai_tools(mcp_tools)
        for question in questions:
            print(f"Q: {question}")
            print(f"A: {await chat_with_tools(llm, mcp_client, tools, question)}\n")


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Answer questions with tools served by an MCP server.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    parser.add_argument("--list-tools", action="store_true", help="list the server's tools and exit")
    args = parser.parse_args(argv)

    asyncio.run(run([args.question] if args.question else SAMPLE_QUESTIONS, args.list_tools))


if __name__ == "__main__":
    main()
