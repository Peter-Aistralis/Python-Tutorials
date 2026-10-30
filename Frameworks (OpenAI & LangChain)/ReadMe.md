# Video 2: Building Agents with OpenAI Agents SDK & LangChain

Companion code for the second video in the "Build Your Own AI Agent" series.

This video does three things:

1. Rebuilds the agent from Video 1 using the **OpenAI Agents SDK** — same loop, now hidden behind `Runner.run`.
2. Rebuilds the same agent using **LangChain** — same abstraction level, different plumbing.
3. Shows **why handoffs fail for a manager workflow**, then fixes it with **Agents as Tools** (`as_tool()`).

> **Important:** `3a_manager_handoff_BROKEN.py` is **intentionally broken**. It exists to demonstrate a real failure mode. The working version is `3b_manager_agents_as_tools_WORKS.py`. See [Handoff vs Agents as Tools](#handoff-vs-agents-as-tools) below for the full explanation.

---

## Files

| File | What it shows | Status |
|------|---------------|--------|
| `1_basic_tool_call_openai_framework.py` | Single agent + two tools using the OpenAI Agents SDK (`Agent`, `function_tool`, `Runner.run`) | ✅ Works |
| `2_basic_tool_call_langgchain_framework.py` | Same agent + tools rewritten with LangChain (`create_agent`, `@tool`, `ChatOpenAI`) | ✅ Works |
| `3a_manager_handoff_BROKEN.py` | Manager workflow attempted with **handoffs** — fails, on purpose | ❌ Broken (intentional) |
| `3b_manager_agents_as_tools_WORKS.py` | Same manager workflow using **Agents as Tools** (`as_tool()`) — works | ✅ Works |

---

## 1. OpenAI Agents SDK — basic tool call

`1_basic_tool_call_openai_framework.py`

```python
from agents import Agent, function_tool, Runner

@function_tool
def get_price(symbol: str) -> float:
    """Get the current price for a cryptocurrency."""
    ...

@function_tool
def send_mail(name: str, message: str) -> str:
    """Send a mail to someone with a message"""
    ...

agent = Agent(
    name="Riri",
    instructions="you are a cute helpful assistant called Riri...",
    tools=[get_price, send_mail],
    model="gpt-4o-mini"
)

result = await Runner.run(agent, "What is the price of bitcoin and Cardano...")
print(result.final_output)
```

The SDK handles the tool-calling loop internally — the same `while True` loop you wrote by hand in Video 1, now hidden inside `Runner.run`.

---

## 2. LangChain — same agent, different framework

`2_basic_tool_call_langgchain_framework.py`

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def get_price(symbol: str) -> float:
    """Get the current price for a cryptocurrency."""
    ...

@tool
def send_mail(name: str, message: str) -> str:
    """Send a mail to someone with a message"""
    ...

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_agent(model, tools, system_prompt="...")

response = await agent_executor.ainvoke({
    "messages": [("user", "What is the price of bitcoin and Cardano...")]
})
```

The abstraction level is similar. The main differences:

- Tools use `@tool` instead of `@function_tool`.
- The agent is created with `create_agent(...)`.
- The result is a message-history dict, not a typed result object.
- Token usage is buried in `response_metadata` / `usage_metadata` per message.

> **Note:** the `system_prompt=` parameter name may vary by LangChain version. If you get a `TypeError`, check the `create_agent` signature in your installed version.

---

## 3a. The handoff attempt — intentionally broken

`3a_manager_handoff_BROKEN.py`

**Goal:** build a "Boss" agent that manages two specialists — Riri (crypto prices) and Peter (email) — using the SDK's handoff mechanism.

**What the code does:** Boss is created with `handoffs=[riri, peter]` so it can transfer control to either specialist.

**What actually happens:** Boss emits **two `transfer_to_*` function calls in the same turn**:

```json
[
  "ResponseFunctionToolCall(name='transfer_to_riri', ...)",
  "ResponseFunctionToolCall(name='transfer_to_peter', ...)"
]
```

Riri takes over, fetches the prices, and returns a final answer that includes:

> *"Sorry, I can't hit 'send' just yet, but you can totally copy this..."*

Peter never sends the email. The task is **half-completed**, and the failure is silent (no error, just a wrong answer).

### Why it fails

A handoff is a **control transfer**, not a function call. When a handoff fires, the receiving agent *takes over the conversation* and becomes the active agent. The SDK can only transfer control to **one** agent at a time — it cannot "call Riri, get a result, then call Peter, get a result, then return to Boss."

The official SDK issue is explicit: *"Handoffs are meant for scenarios where you transfer control of the entire conversation to a new agent — so it's not possible to hand off to multiple agents."*

So when Boss emits two `transfer_to_*` calls in one turn, only one can take effect. The second is orphaned. That's exactly what the log shows.

**This is not a bug in the script.** It is the script demonstrating the wrong tool for the job.

---

## 3b. The workaround — Agents as Tools

`3b_manager_agents_as_tools_WORKS.py`

Replace `handoffs=[...]` with `agent.as_tool(...)`:

```python
riri = Agent(
    name="Riri",
    instructions="You are Riri, a specialist for cryptocurrency prices.",
    tools=[get_price],
    model="gpt-4o-mini"
)

peter = Agent(
    name="Peter",
    instructions="You are Peter, a specialist for sending emails.",
    tools=[send_mail],
    model="gpt-4o-mini"
)

boss = Agent(
    name="Boss",
    instructions=(
        "You are Boss, an intelligent manager. "
        "You have access to specialist agents as tools. "
        "Call them as needed, collect their results, and produce the final answer yourself."
    ),
    tools=[
        riri.as_tool(
            tool_name="get_crypto_price",
            tool_description="Get cryptocurrency prices from Riri."
        ),
        peter.as_tool(
            tool_name="send_email_via_peter",
            tool_description="Send an email via Peter."
        )
    ],
    model="gpt-4o-mini"
)

result = await Runner.run(boss, "What is the price of bitcoin and Cardano. send both price by mail to Peter")
```

Now Boss:

1. Calls `get_crypto_price` → gets BTC and ADA prices.
2. Calls `send_email_via_peter` → sends the email.
3. Produces the final answer itself.

**Boss stays in control the whole time.** The specialists are bounded capabilities, not conversation owners. The task completes.

---

## Handoff vs Agents as Tools

This is the core distinction of Video 2.

| | Handoff | Agents as Tools |
|---|---|---|
| Who owns the final answer | The specialist takes over | The manager keeps control |
| Control flow | Transfer (one-way) | Call and return |
| Can call multiple specialists per turn | ❌ No | ✅ Yes |
| Good for | "This specialist should answer the user now" | "This specialist should do a bounded job and report back" |
| OpenAI SDK syntax | `Agent(handoffs=[...])` or `handoff(agent)` | `agent.as_tool(...)` |

**Rule of thumb:**

> If the manager needs to **collect results from multiple specialists and synthesize a final answer**, use **Agents as Tools**.  
> If a specialist should **own the rest of the conversation**, use **handoffs**.

Your Boss/Riri/Peter workflow is the first case. The `handoff` import in `3a` was intentional — it shows the wrong tool, on purpose.

---

## LangChain and handoffs (for the record)

LangChain v1 **does** support multi-agent handoffs, but through a different mechanism than the OpenAI SDK:

- `HandoffAgentMiddleware` for middleware-based agent switching.
- LangGraph `Command` for explicit state transitions between agents.

If you want to demonstrate handoffs in LangChain in a future video, start there — not with the OpenAI SDK's `as_tool()` syntax, and not by assuming the two frameworks share a handoff API.

The "handoff doesn't work for this workflow" lesson applies to **any** framework: handoffs are control transfers, not RPC calls. The fix (a manager calling specialists as tools) is framework-agnostic.

---

## Running

```bash
pip install openai-agents langchain langchain-openai python-dotenv requests
```

Create `.env` in the repo root:

```
Openai_Key=sk-...
CMC_Key=...
```

Run any file:

```bash
python 1_basic_tool_call_openai_framework.py
python 2_basic_tool_call_langgchain_framework.py
python 3a_manager_handoff_BROKEN.py       # observe the failure
python 3b_manager_agents_as_tools_WORKS.py # observe the fix
```

> `nest_asyncio.apply()` is included for Spyder/Jupyter environments only. It is not part of the agent logic.

---

## What to watch for

In `3a_manager_handoff_BROKEN.py`, look at the `raw_responses` dump:

```json
[
  { "output": ["transfer_to_riri", "transfer_to_peter"] },
  { "output": ["get_price(BTC)", "get_price(ADA)"] },
  { "output": ["message: Sorry, I can't hit 'send' just yet..."] }
]
```

Two things to notice:

1. **Turn 1 has two `transfer_to_*` calls.** Only one can take effect.
2. **The final message comes from Riri, not Boss.** Control was transferred, not borrowed.

In `3b_manager_agents_as_tools_WORKS.py`, the equivalent dump shows Boss calling `get_crypto_price`, then `send_email_via_peter`, then producing the final message itself.

---

## Takeaways

1. **The framework didn't do anything you didn't do in Video 1.** `Runner.run` and `create_agent` are the same two-loop ReAct structure, wrapped in ergonomics.
2. **Handoffs ≠ function calls.** A handoff transfers conversation control to one agent. It is not "call and return."
3. **Agents as Tools is the manager pattern.** `agent.as_tool()` lets a manager call specialists, collect results, and stay in control.
4. **The failure is the lesson.** `3a` is intentionally broken so you can see *why* the wrong tool produces a silent half-completed task — not a stack trace, just a polite "sorry, I can't hit send."
