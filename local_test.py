from openai import OpenAI

# We point the standard OpenAI client at your local machine!
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # The key doesn't matter for local execution
)

print("Thinking...")
response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a highly concise medical AI."},
        {
            "role": "user",
            "content": "Explain what a CNN is in computer vision in one sentence.",
        },
    ],
)

print("\nResponse:")
print(response.choices[0].message.content)
