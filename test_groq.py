import os

from groq import Groq


api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")


client = Groq(api_key=api_key)


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain what RAG is in one simple sentence.",
        }
    ],
)


answer = response.choices[0].message.content

print("Groq response:")
print(answer)