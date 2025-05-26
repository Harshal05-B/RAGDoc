import requests
import yaml
from pathlib import Path

# Load plugin YAML config
def load_mistral_config(yaml_path: str = "config.yaml") -> dict:
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract base URL and API key from server and security sections
    base_url = data.get("servers", [{}])[0].get("url", "https://api.mistral.ai")
    api_key = data.get("securitySchemes", {}).get("ApiKey", {}).get("key", None)
    model = "mistral-large-latest"  # fallback, override later if needed
    return {"api_key": api_key, "base_url": base_url, "model": model}


# Send a chat completion request
def get_llm_response(context: str, query: str) -> str:
    config = load_mistral_config("plugin-redoc-0.yaml")  # Adjust if renamed
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]

    endpoint = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] Mistral API call failed: {e}"
