import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2:3b",
    "prompt": "Explain Python in simple words.",
    "stream": False
}

response = requests.post(url, json=data)

result = response.json()

print("\n==============================")
print("OLLAMA RESPONSE")
print("==============================")

print(result["response"])