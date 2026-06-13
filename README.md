# KiteFishAI Python SDK

Official Python SDK for [KiteFishAI](https://kitefishai.com) — sovereign AI models for regulated Indian enterprises.

## Installation

```bash
pip install kitefishai
```

## Quick Start

```python
import kitefishai

client = kitefishai.KiteFishAI(api_key="kf-...")

# Or use environment variable: KITEFISH_API_KEY
```

## Chat

```python
# Non-streaming
response = client.chat.complete(
    model="kf-reasoning-10b",
    messages=[
        {"role": "user", "content": "Summarise the DPDP Act 2023."}
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

```python
# Streaming
with client.chat.stream(
    model="kf-reasoning-10b",
    messages=[{"role": "user", "content": "Explain claim settlement process."}],
) as stream:
    for chunk in stream:
        print(chunk.delta, end="", flush=True)

full_text = stream.get_final_text()
```

```python
# With system prompt
response = client.chat.complete(
    model="kf-reasoning-10b",
    system="You are a BFSI compliance assistant. Respond in formal English.",
    messages=[{"role": "user", "content": "What are KYC requirements?"}],
)
```

## Embeddings

```python
result = client.embeddings.create(
    model="minnow-em-v1",
    input=["query: insurance renewal process", "passage: Renewal can be done..."],
)

for item in result.data:
    print(f"[{item.index}] dim={len(item.embedding)}")
```

```python
# MRL — request smaller dimensions (896 / 512 / 256 / 128 / 64)
result = client.embeddings.create(
    model="minnow-em-v1",
    input="query: what is DPDP?",
    dimensions=256,
)
```

## Error Handling

```python
import kitefishai

try:
    response = client.chat.complete(model="kf-reasoning-10b", messages=[...])
except kitefishai.AuthenticationError:
    print("Invalid API key")
except kitefishai.RateLimitError:
    print("Rate limit hit — back off and retry")
except kitefishai.APIError as e:
    print(f"API error {e.status_code}: {e.message}")
except kitefishai.KiteFishAIError as e:
    print(f"SDK error: {e}")
```

## Context Manager

```python
with kitefishai.KiteFishAI(api_key="kf-...") as client:
    response = client.chat.complete(...)
# Connection closed automatically
```

## Configuration

| Parameter     | Default                          | Description                        |
|---------------|----------------------------------|------------------------------------|
| `api_key`     | `KITEFISH_API_KEY` env var       | Your API key                       |
| `base_url`    | `https://api.kitefishai.com/v1`  | Override for on-prem deployments   |
| `timeout`     | `60.0`                           | Request timeout in seconds         |
| `max_retries` | `2`                              | Retries on timeout/network errors  |

```python
# On-prem / air-gapped deployment
client = kitefishai.KiteFishAI(
    api_key="kf-...",
    base_url="https://your-internal-host/v1",
    timeout=120.0,
)
```

## Requirements

- Python 3.9+
- `httpx >= 0.25.0`

## License

MIT — see [LICENSE](LICENSE)