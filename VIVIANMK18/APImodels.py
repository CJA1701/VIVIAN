"""List available Anthropic models. Reads ANTHROPIC_API_KEY from env,
falling back to config.yaml."""

import os

import anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    try:
        import yaml
        with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
            api_key = (yaml.safe_load(f) or {}).get("anthropic", {}).get("api_key")
    except Exception:
        pass

if not api_key:
    raise RuntimeError("No Anthropic API key in ANTHROPIC_API_KEY or config.yaml")

client = anthropic.Anthropic(api_key=api_key)

print("Available models:\n")
for model in client.models.list():
    print(f"{model.id:30s} {model.display_name}")
