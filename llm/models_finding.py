import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
  print("[ERROR] GROQ_API_KEY not found in your environment or .env file.")
  exit(1)

print("Connecting to Groq API to fetch available models...\n")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

try:
  models = client.models.list()
  print("--- Supported Models on Your Groq API Key ---")
  for model in models.data:
    print(f"• ID: {model.id}")
except Exception as e:
  print(f"[ERROR] Failed to fetch models: {e}")