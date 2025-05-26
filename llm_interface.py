import os
from dotenv import load_dotenv
from together import Together

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
print("API Key loaded successfully.",api_key)

if not api_key:
    raise EnvironmentError("TOGETHER_API_KEY not found in .env file.")

# Initialize Together client
client = Together(api_key=api_key)

def get_llm_response(context: str, query: str, model: str = "deepseek-ai/DeepSeek-V3") -> str:
    """
    Sends a chat completion request to Together API and returns the full response.
    Streaming output is printed live.
    """
    prompt = f"""Answer the question based on the following context.

Context:
{context}

Question:
{query}

Answer:"""

    messages = [{"role": "user", "content": prompt}]

    # Use stream=True to print while generating
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    collected = []
    for token in response:
        if hasattr(token, "choices"):
            delta = token.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                collected.append(delta)
    return "".join(collected).strip()
