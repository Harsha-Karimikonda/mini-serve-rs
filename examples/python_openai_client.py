#!/usr/bin/env python3
"""
Python OpenAI SDK Drop-in Example for Mini-Serve

Demonstrates using Mini-Serve as a drop-in replacement for the official OpenAI API
in Python. Works seamlessly with existing OpenAI, LangChain, or LlamaIndex workflows.

Requirements:
    pip install openai

Run:
    python examples/python_openai_client.py
"""

import sys
try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' package is required. Install it with: pip install openai")
    sys.exit(1)

def main():
    # 1. Initialize OpenAI client pointing to Mini-Serve
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed"  # Mini-Serve does not require auth by default
    )

    print("=" * 60)
    print("⚡ Mini-Serve Python Client (OpenAI SDK Drop-in)")
    print("=" * 60)

    # 2. List Models
    print("\n--- 1. Querying /v1/models ---")
    try:
        models = client.models.list()
        for m in models.data:
            print(f"Available Model: id={m.id}, owned_by={m.owned_by}")
    except Exception as e:
        print(f"Error connecting to Mini-Serve: {e}")
        print("Please ensure Mini-Serve is running on http://127.0.0.1:8000")
        return

    # 3. Non-streaming Chat Completion
    print("\n--- 2. Non-Streaming Chat Completion ---")
    response = client.chat.completions.create(
        model="mock",
        messages=[
            {"role": "system", "content": "You are a helpful software engineering assistant."},
            {"role": "user", "content": "Why is Rust popular for high-performance systems?"}
        ],
        temperature=0.7,
        max_tokens=60
    )
    print(f"Assistant: {response.choices[0].message.content}")
    print(f"Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")

    # 4. Streaming Chat Completion
    print("\n--- 3. Real-Time Streaming Chat Completion ---")
    print("Assistant > ", end="", flush=True)
    stream = client.chat.completions.create(
        model="mock",
        messages=[
            {"role": "user", "content": "Explain continuous batching in one paragraph."}
        ],
        stream=True,
        max_tokens=80
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n\n" + "=" * 60)
    print("✓ Successfully executed all Python use cases with zero custom SDK code!")
    print("=" * 60)

if __name__ == "__main__":
    main()

