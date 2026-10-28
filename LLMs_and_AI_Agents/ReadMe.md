# Building an AI Agent from Scratch

A step-by-step progression from a plain LLM call to a full ReAct agent — using nothing but the OpenAI Responses API and plain Python. No agent frameworks required.

The series is deliberately built as **five scripts that each add exactly one idea**, and — more importantly — **each exposes a specific bug class** that only appears once you go one level deeper. By step 5 you have the canonical agent loop that every framework (including `openai-agents`) wraps for you.

## The five steps

| # | File | Concept | Assumption it makes | What breaks it |
|---|------|---------|---------------------|----------------|
| 1 | `1_basic_llm_call.py` | Plain LLM call | — | — |
| 2 | `2_basic_tool_call.py` | One tool call, one turn | "there is exactly one `function_call`" | — (works for this prompt) |
| 3 | `3_sequential_tool_calling.py` | Multiple tools, one call per turn | "one call per turn" | A prompt that makes the model batch calls |
| 4 | `4_parallel_tool_calling.py` | Multiple calls in **one** `response.output` | "one turn is enough" | A task that needs a **second** turn (e.g. email *after* fetching price) |
| 5 | `5_react_(full_working_ai_agent).py` | **Both loops** — N calls per turn × N turns per task | — (general case) | Nothing — this is the full ReAct loop |

The important progression: **steps 3 and 4 each fix one loop**, and **step 5 is just both loops at once** — which is the whole agent.

---

## 1. Basic LLM call — `1_basic_llm_call.py`

The baseline. Send a system + user message, read the text back.

```python
response = client.responses.create(model=model_name, input=base_message)
print(response.output_text)
```

Nothing agentic yet. This exists so the viewer has a "before" picture and sees how little code is needed to get a model response.

---

## 2. Your first tool call — `2_basic_tool_call.py`

Introduce a single tool (`get_price`) and the function-calling protocol:

1. Send tools + messages.
2. Model returns a `function_call` instead of a message.
3. You execute the function and send the result back.
4. Model returns the final text.

```python
if response_llm.output[0].type == "function_call":
    tool_call_object = response_llm.output[0]
    args = json.loads(tool_call_object.arguments)
    price = get_price(args["symbol"])
    # ... feed the result back and ask again
```

⚠️ **This works — but only because it assumes `response.output[0]` is the only tool call.** That assumption is invisible here and will bite in step 3.

> Note: `2_basic_tool_call_framework.py` is the same step rewritten using the `openai-agents` SDK (`Agent`, `function_tool`, `Runner`). It's included to show what the framework *does* for you — and to make the point that steps 3–5 are what's happening under the hood of `Runner.run`.

---

## 3. Sequential tool calling — `3_sequential_tool_calling.py`

Add a second tool (`send_email`) and a `while` loop so the agent can do *multiple* rounds:

```
Turn 1: [get_price(BTC)]
Turn 2: [send_email(Peter, "...")]
Turn 3: message("Done! 🎉")
```

The loop looks reasonable:

```python
for item in response.output:
    if item.type == "function_call":
        ... execute ...
        base_message.append({... "function_call_output" ...})
        response = client.responses.create(...)   # re-ask
        break                                      # ⬅ exits after ONE call
```

✅ This works **for truly sequential prompts** — one call per turn, one turn at a time. That is a real class of tasks, and the script is correct for them.

### 🐛 The bug it hides

The moment the model emits **more than one `function_call` in a single `response.output`**, this loop executes the first one, re-asks, and `break`s — the remaining calls are never executed and never get a `function_call_output`. The API then rejects the next request:

```
BadRequestError: 400 — No tool output found for function call call_xxx.
```

The tell is the print below. Dump the items in the turn:

```python
print([i.type for i in response.output])
# ['function_call', 'function_call']   ← two calls in one turn
```

`status='completed'` on both is **normal** — it describes the item's lifecycle, not whether *you* executed it.

---

## 4. Parallel tool calling — `4_parallel_tool_calling.py`

Fixes the inner loop: **execute every `function_call` in the turn before re-asking.**

```python
for tool_call in response.output:
    if tool_call.type != "function_call":
        continue
    result = call_function(tool_call.name, json.loads(tool_call.arguments))
    base_message.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": json.dumps(result),
    })
# re-ask ONCE, outside the loop
response = client.responses.create(model="gpt-4o-mini", input=base_message, tools=tools)
```

Now a turn like `[get_price(BTC), get_price(ADA)]` works: both get executed, both outputs are appended, one re-ask follows.

### 🐛 The bug it hides

This script does **exactly one** tool-execution round. If the task needs a second round — e.g. *"fetch BTC price, **then** email it to Peter"* — the model is still mid-task after round 1. The script prints `response.output_text`, but `response.output` contains a `function_call`, not a message, so the text is empty:

```
🔥 FINAL RIRI ANSWER 🔥
                        ← nothing
```

The `send_email` call requires the price result, so it can never appear in turn 1. The final message only comes after **turn 2**'s tool output is appended and you re-ask once more.

---

## 5. The full ReAct agent — `5_react_(full_working_ai_agent).py`

Both loops, together. This is **ReAct** (Reasoning + Acting) implemented with native function calling:

```
Turn 1: [get_price(BTC), get_price(ADA)]                   ← parallel lookups
Turn 2: [send_email(Peter, BTC), send_email(XXX, ADA)]     ← parallel, dependent on turn 1
Turn 3: message("Bitcoin is $X, Cardano is $Y — emails sent! 🎉")
```

```python
while True:
    base_message += response.output                       # append this turn ONCE

    function_calls = [i for i in response.output if i.type == "function_call"]

    if not function_calls:                                # model is done
        print(response.output_text)
        break

    for tool_call in function_calls:                      # execute ALL calls in the turn
        result = call_function(tool_call.name, json.loads(tool_call.arguments))
        base_message.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result),
        })

    response = client.responses.create(                   # re-ask ONCE
        model=model_name, input=base_message, tools=tools
    )
```

### The invariant

> **Every `function_call` gets exactly one `function_call_output`; every turn's `response.output` is appended exactly once; re-ask until there are no more `function_call`s.**

- Inner `for` loop → **N calls per turn** (step 4's fix).
- Outer `while` loop → **N turns per task** (step 5's new piece).
- They are orthogonal. You need both.

Once this clicks, the `openai-agents` `Runner.run` from `2_basic_tool_call_framework.py` stops being magic — it *is* this loop, with retries, streaming, and tracing wrapped around it.

---

## Bug taxonomy

| Step | Assumption | Symptom |
|------|------------|---------|
| 2 | "one call, one turn" | none (happy path) |
| 3 | "one call per turn" | `400 No tool output found for function call ...` |
| 4 | "one turn is enough" | empty final answer / task silently incomplete |
| 5 | — | none |

The reassuring part: **every bug in this series is the same mistake** — an unstated assumption about *how many* calls or *how many* turns there are. The fix is never new code; it's removing the assumption. The model decides how to batch calls based on the prompt, and the prompt wording *does not guarantee* the batching:

- *"price of bitcoin and send it to Peter"* → often **sequential** (1 call/turn) → step 3 works.
- *"price of bitcoin and cardano, email BTC to Peter and ADA to XXX"* → often **parallel** (2 calls/turn) → step 3 dies, step 4 works — but step 4 dies on the multi-round part.

That's why step 5 is the target: prompts describe *intent*, not *execution shape*. The agent loop must be agnostic to both.

---

## Running the scripts

```bash
pip install openai requests python-dotenv
# optional, only for 2_basic_tool_call_framework.py:
pip install openai-agents nest-asyncio
```

Create a `.env` in the repo root:

```
Openai_Key=sk-...
CMC_Key=...        # https://coinmarketcap.com/api/
```

Then run any step:

```bash
python 1_basic_llm_call.py
python 2_basic_tool_call.py
python 3_sequential_tool_calling.py
python 4_parallel_tool_calling.py
python "5_react_(full_working_ai_agent).py"
```

> **Watch for:** `400 No tool output found for function call ...` (step 3 on a parallel prompt) and an empty `🔥 FINAL RIRI ANSWER 🔥` block (step 4 on a multi-round prompt). Those are the two bugs the series is built to demonstrate.

---

## Takeaways

1. **`response.output` is a list, not an item.** It can contain a message, one `function_call`, several `function_call`s, or a mix.
2. **`status='completed'` ≠ "I executed it."** It's the item's lifecycle status on the API side. Your job is to give every `function_call` a matching `function_call_output` (same `call_id`).
3. **Prompt wording ≠ execution shape.** "First do A, then B" can still produce parallel calls in one turn. Don't encode prompt semantics into your loop structure.
4. **Two orthogonal loops.** Inner loop handles N calls per turn; outer loop handles N turns per task. Steps 3 and 4 each teach one; step 5 is both.
5. **That's ReAct.** Think → act → observe → repeat, using native function calling instead of prompt parsing. Frameworks like `openai-agents` are this loop with ergonomics on top.

---

*If this was useful, the companion video walks through each script, the two error messages, and the "aha" moment where the two loops snap together.*
