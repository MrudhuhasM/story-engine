"""test_llm.py — minimal connectivity test before running the full pipeline.

Run this first:
  python test_llm.py

If this works, the full pipeline will work.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)

print("Testing non-streaming...")
response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "user", "content": "Say exactly: working"}
    ],
    max_tokens=10,
    temperature=0.1,
)
print("Non-streaming response:", repr(response.choices[0].message.content))

print("\nTesting streaming...")
stream = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "user", "content": "Count to 5."}
    ],
    max_tokens=30,
    temperature=0.1,
    stream=True,
)
for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)

print("\n\nBoth tests passed.")