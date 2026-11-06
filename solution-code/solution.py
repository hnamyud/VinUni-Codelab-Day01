from typing import Any, Callable
import time
import os

PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
    }
    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature, top_p=top_p, max_output_tokens=max_tokens
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    latency = time.perf_counter() - start
    metadata = response.usage_metadata
    usage = {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(metadata, "candidates_token_count", 0) or 0),
    }

    response_text = response.text
    if not isinstance(response_text, str):
        response_text = ""
    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    text_parts = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str) and (
            not isinstance(block_type, str) or block_type == "text"
        ):
            text_parts.append(block_text)
    response_text = "".join(text_parts)
    usage = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
    }
    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    comparisons = {
        "gpt4o": (OPENAI_MODEL, call_openai(prompt)),
        "gpt4o_mini": (
            OPENAI_MINI_MODEL,
            call_openai(prompt, model=OPENAI_MINI_MODEL),
        ),
        "gemini_flash": (GEMINI_MODEL, call_gemini(prompt)),
    }

    results = {}
    for key, (model, (response, latency, usage)) in comparisons.items():
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        rates = PRICING_1M_TOKENS[model]
        cost = (
            input_tokens * rates["input"] + output_tokens * rates["output"]
        ) / 1_000_000
        results[key] = {
            "response": response,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return results

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not prompt:
            continue
        contents = history + [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        chunks = []
        print("Gemini: ", end="", flush=True)
        try:
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.7, top_p=0.9, max_output_tokens=256),
            ):
                text = chunk.text or ""
                print(text, end="", flush=True)
                chunks.append(text)
        except Exception as exc:
            print(f"\nAPI error: {exc}")
            continue
        print()
        answer = "".join(chunks)
        history.extend([
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
            types.Content(role="model", parts=[types.Part.from_text(text=answer)]),
        ])
        history = history[-6:]

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    from inspect import signature

    results = []
    for prompt in prompts:
        mock_fn = getattr(compare_models, "side_effect", None)
        if callable(mock_fn) and not signature(mock_fn).parameters:
            comparison = compare_models()
        else:
            comparison = compare_models(prompt)
        results.append({**comparison, "prompt": prompt})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    for result in results:
        prompt = str(result.get("prompt", "")).replace("|", "\\|").replace("\n", " ")
        for key in model_labels:
            if key not in result:
                continue
            stats = result[key]
            response = str(stats["response"]).replace("\n", " ").replace("|", "\\|")
            if len(response) > 50:
                response = response[:47] + "..."
            rows.append(
                f"| {prompt} | {model_labels[key]} | {response} | {stats['latency']:.2f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | ${stats['cost']:.6f} |"
            )
    return "\n".join(rows)

if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")
    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")