"""test_raw_stream.py — inspect every field of every streaming chunk.

Run: uv run tests/test_raw_stream.py

This tells us exactly where thinking tokens are — content, reasoning_content,
or inside <think> tags in content.
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local", timeout=120.0)

print("=== Raw streaming chunk inspection (first 30 chunks) ===\n")

stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What is 2+2? Think step by step."}],
    max_tokens=300,
    temperature=0.1,
    stream=True,
)

for i, chunk in enumerate(stream):
    if i >= 30:
        print("\n... (truncating at 30 chunks)")
        break

    delta = chunk.choices[0].delta

    # What fields does delta actually have?
    delta_dict = delta.__dict__ if hasattr(delta, "__dict__") else {}

    content = getattr(delta, "content", None)
    reasoning = getattr(delta, "reasoning_content", None)  # llama.cpp thinking field
    reasoning_alt = getattr(delta, "reasoning", None)       # alternate name

    # Only print chunks that have something
    if content or reasoning or reasoning_alt:
        print(f"chunk {i:02d}:")
        if content is not None:
            print(f"  content          = {repr(content)}")
        if reasoning is not None:
            print(f"  reasoning_content= {repr(reasoning)}")
        if reasoning_alt is not None:
            print(f"  reasoning        = {repr(reasoning_alt)}")

        # Print any other non-None fields we might have missed
        for k, v in delta_dict.items():
            if k not in ("content", "reasoning_content", "reasoning", "role") and v is not None:
                print(f"  {k} = {repr(v)}")

print("\n=== Done ===")
print("\nConclusion: look at which field carries the <think> tokens above.")