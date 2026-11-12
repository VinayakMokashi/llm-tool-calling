# LLM Tool Calling

Four small, runnable programs that show how an LLM uses tools through **native tool calling**: the model returns a structured request to call a function, your code runs the function, and the result goes back to the model. They start with one tool and one round, add a loop for multi-step problems, apply it to an inventory assistant, and end with the tools served from a separate **MCP (Model Context Protocol)** server.

Built with [LangChain](https://python.langchain.com/), [FastMCP](https://gofastmcp.com/) and [Groq](https://groq.com/)-hosted `gpt-oss` models.

| # | Script | What it shows |
| --- | --- | --- |
| 1 | [`01_single_tool.py`](01_single_tool.py) | One tool, one round. The model decides whether a question needs the tool at all. |
| 2 | [`02_multi_step_tools.py`](02_multi_step_tools.py) | A calculator with six tools and a loop, so one tool's output can become the next tool's input. |
| 3 | [`03_inventory_tools.py`](03_inventory_tools.py) | An inventory assistant: the model picks among four tools and fills in their arguments from plain English. |
| 4 | [`04_mcp/`](04_mcp/) | The tools live in an MCP server process; the client discovers them at run time and passes the model's calls through MCP. |

## How tool calling works

```
            question + tool descriptions (name, docstring, argument schema)
your code ───────────────────────────────────────────────────────────────► LLM
    ▲                                                                       │
    │      "call square_root with {"number": 2025}"  (a structured request) │
    ◄───────────────────────────────────────────────────────────────────────┘
    │
    ├── runs square_root(2025) ─► 45.0
    │
    │      the result, as a tool message linked to the request's id
    └──────────────────────────────────────────────────────────────────────► LLM ─► "The square root of 2025 is 45."
```

- **The model never runs code.** It only asks for a tool by name, with arguments that match the tool's schema. Your code decides whether and how to run it.
- **The tool description is a prompt.** LangChain's `@tool` and FastMCP's `@mcp.tool()` build each tool's description from the function's name, type hints and docstring. The docstring is all the model knows about when and how to use the tool.
- **The model chooses.** With `tool_choice="auto"`, the default, a question that needs no tool ("What is the capital of France?") is answered directly.
- **Results are linked to requests.** Each tool result is sent back as a tool message carrying the `id` of the call it answers, after the model's own message that made the request.

## The scripts

The sample outputs below come from real runs, shortened where an answer was long.

### 1. Single tool, single round

`square_root` is the only tool. `chat_with_tools()` sends the question, runs any tool the model asks for, and sends one follow-up request for the answer. A failing tool (for example a negative number) returns its error as text, so the model can explain it instead of the program crashing.

```
Q: What is the capital of France?
A: The capital of France is **Paris**.

Q: Can you calculate the square root of 2025 for me?
  tool: square_root({'number': 2025}) -> 45.0
A: The square root of 2025 is **45**.
```

One round is the limit of this design: if the model needs a second tool after seeing the first result, this script stops and says so.

### 2. Multi-step tool calling

Six arithmetic tools (`add`, `multiply`, `subtract`, `divide`, `square_root`, `power`) and a system prompt that tells the model to use a tool for every step. The loop keeps going until the model answers without asking for a tool, or `max_turns` is reached:

```
ask the model ─► tool calls? ── no ──► final answer
      ▲               │ yes
      │               ▼
      └── send results back ◄── run every requested tool
```

This is the ReAct pattern (reason, act, observe) on top of native tool calling. The model must chain three tools in order here, because each step needs the previous result:

```
Q: What is the square root of the product of the sum of 2 and 3 with 12.5?
  [turn 1] add({'numbers': [2, 3]}) -> 5.0
  [turn 2] multiply({'numbers': [5, 12.5]}) -> 62.5
  [turn 3] square_root({'number': 62.5}) -> 7.905694150420948
A: The result is **7.905694150420948**.
```

### 3. Inventory assistant

Five SKUs are held in memory with stock, reorder point, unit cost and average daily demand. Four tools answer questions about them:

| Tool | Answers |
| --- | --- |
| `get_stock_level(sku)` | Stock, reorder point and LOW/OK status for one SKU |
| `list_low_stock_items()` | Every SKU at or below its reorder point |
| `days_until_stockout(sku)` | Stock divided by daily demand |
| `calculate_reorder_quantity(sku, lead_time_days=7)` | Units needed to cover two lead times of demand (one for the wait, one as safety stock), with the order cost |

The model maps plain English onto these calls, including arguments: "my supplier takes 10 days" becomes `lead_time_days=10`. With the loop from script 2 it can also go further than the question strictly asks. Asked only which items need reordering, it listed them and then sized an order for each:

```
Q: Which items need to be reordered right now?
  tool: list_low_stock_items({})
  tool: calculate_reorder_quantity({'lead_time_days': 7, 'sku': 'SKU-002'})
  tool: calculate_reorder_quantity({'lead_time_days': 7, 'sku': 'SKU-003'})
A: | SKU | Item | Current Stock | Reorder Point | Recommended Reorder Qty | Estimated Cost |
   | SKU-002 | Hydraulic Pump | 18 | 25 | 24 units | $10,320.00 |
   | SKU-003 | Conveyor Belt | 5 | 10 | 23 units | $4,847.25 |
```

Every number in that answer comes from a tool, so it can be checked: SKU-002 sells 3 a day, so two 7-day lead times need 42 units, minus the 18 in stock is 24.

### 4. Tools over MCP

In scripts 1-3 the tools sit in the same file as the agent. MCP puts a standard interface between them: an **MCP server** publishes tools, and any **MCP client** can discover and call them, so a tool written once works in every MCP-compatible app.

```
client.py ──── starts ────► server.py (separate process, talks MCP over stdin/stdout)
    │   tools/list ──────────►   get_current_time(timezone)
    │   ◄── names, descriptions, JSON Schemas   get_weather(city)  ── Open-Meteo API
    │
    ├── converts the tools to the OpenAI function format and asks the model (gpt-oss-20b on Groq)
    │
    │   tools/call get_weather {"city": "Bengaluru"} ──►
    │   ◄── "Weather in Bengaluru, Karnataka, India: Overcast, 19.7°C ..."
    │
    └── sends the result back to the model ─► final answer
```

- **[`server.py`](04_mcp/server.py)** defines two tools with `@mcp.tool()`: `get_current_time` (Python's time zone database) and `get_weather`, which calls the free [Open-Meteo](https://open-meteo.com/) API in two steps, city name to coordinates and then coordinates to current conditions. Both return problems as text instead of raising.
- **[`client.py`](04_mcp/client.py)** starts the server as a subprocess, lists its tools, and copies each tool's JSON Schema straight into the OpenAI function-calling format (both use JSON Schema). It then runs the same loop as script 2, except that each tool call goes to the server through MCP. The model is called with the official `openai` SDK pointed at Groq's OpenAI-compatible endpoint.

The model never talks to MCP; the client is the bridge. The model reads the docstrings, too. `get_weather` asks for official city names because the geocoder may not find "Bangalore", and the time tool's example names `Asia/Kolkata`:

```
Q: What time is it in Bangalore?
  tool via MCP: get_current_time({'timezone': 'Asia/Kolkata'}) -> Current time in Asia/Kolkata: 2026-11-09 02:26:24 IST
A: It's 2:26 am (02:26) on 9 November 2026 in Bangalore (Asia/Kolkata, Indian Standard Time).

Q: What's the weather like in Bangalore?
  tool via MCP: get_weather({'city': 'Bengaluru'}) -> Weather in Bengaluru, Karnataka, India: Overcast, 19.7°C (feels like 21.3°C), Humidity: 93%, Wind: 11.0 km/h
A: The current weather in Bengaluru (Bangalore) is: Overcast, 19.7 °C (feels like 21.3 °C), humidity 93 %, wind 11.0 km/h.
```

## Repository structure

```
llm-tool-calling/
├── 01_single_tool.py        # one tool, one round of tool calls
├── 02_multi_step_tools.py   # calculator: a loop that chains tool results
├── 03_inventory_tools.py    # inventory assistant with four tools
├── 04_mcp/
│   ├── server.py            # FastMCP server: time and weather tools (stdio)
│   └── client.py            # MCP client that hands the server's tools to the model
├── requirements.txt
├── .env.example             # template for your Groq API key
└── LICENSE
```

## Tech stack

- Python 3.10+
- `langchain-core` and `langchain-groq` 1.x (scripts 1-3): the `@tool` decorator, `bind_tools` and the Groq chat model
- `fastmcp` 4.x (script 4): MCP server and client
- `openai` (script 4): the official SDK, pointed at Groq's OpenAI-compatible endpoint
- `requests` (the weather tool) and `tzdata` (time zones on Windows)
- `python-dotenv` (loads your API key from `.env`)
- Groq-hosted models: `openai/gpt-oss-120b` (scripts 1-3) and `openai/gpt-oss-20b` (script 4)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/llm-tool-calling.git
   cd llm-tool-calling
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key.** Copy `.env.example` to `.env` and paste in your key (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Every script loads `.env` from the repository root, so the key is found no matter which directory you run from. `.env` is git-ignored and never committed.

## How to run

```bash
python 01_single_tool.py
python 02_multi_step_tools.py
python 03_inventory_tools.py
python 04_mcp/client.py
```

With no arguments, each script answers its sample questions. Every script also takes `--question`:

```bash
python 02_multi_step_tools.py --question "What is 2 to the power of 10, divided by 4?"
python 03_inventory_tools.py --question "How many days of stock do we have for SKU-005?"
python 04_mcp/client.py --question "Is it raining in Mumbai right now?"
```

| Option | Scripts | Description |
| --- | --- | --- |
| `--question` | all | Ask your own question instead of the samples |
| `--max-turns` | 2 | Maximum rounds of tool calls (default 20) |
| `--list-tools` | 4 | Print the MCP server's tools and their schemas, without calling the model or needing an API key |

The client starts the server itself. When it does, the server logs one line, `Starting MCP server 'Time and Weather' with transport 'stdio'`, before the answers. You can also start the server on its own with `python 04_mcp/server.py`, but it then just waits for MCP messages on stdin.

Answers vary a little from run to run; the tool calls are usually the same.

## Limitations

- **The model can skip a tool.** Nothing forces a tool call. The calculator's system prompt says to use tools for every step, but a model can still do arithmetic itself.
- **The agent can only do what its tools allow.** The inventory tools look items up by SKU. Asked about "the pressure gauges" by name, the model asks for the SKU instead of guessing, because no tool searches by name. Unknown SKUs are reported as not found.
- **Sample data only.** The inventory lives in a Python dictionary, and its reorder rule (two lead times of demand) is a simple heuristic, not a demand model.
- **Local server only.** The MCP server runs over stdio as a child process. Serving it to remote clients would need MCP's HTTP transport and authentication.
- **No memory.** Each question starts a new conversation.

## See also

[agentic-patterns-from-scratch](https://github.com/VinayakMokashi/agentic-patterns-from-scratch) builds the same tool-use and ReAct patterns without native tool calling: the tools are described in the prompt and the model's calls are parsed out of its text. Comparing the two shows what native tool calling takes off your hands.

[llm-agent-memory](https://github.com/VinayakMokashi/llm-agent-memory) adds the memory these agents lack: keeping a long conversation within limits (trimming, deletion and summarization), and long-term memory that an agent reads and updates itself.

## License

Released under the [MIT License](LICENSE).
