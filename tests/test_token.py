import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Load the environment variables
load_dotenv()
token = os.environ.get("HF_TOKEN")

print("========================================")
print("🔍 NUST Bank Token Diagnostic Test")
print("========================================")

# 2. Check if the token is actually loading from .env
if not token:
    print("❌ ERROR: No token found. Your .env file is not being read properly.")
    exit()

# Print the first and last few characters to verify it's the RIGHT token
print(f"✅ Token loaded into memory: {token[:4]}.....{token[-4:]}")

# 3. Test the Inference API permissions
print("⏳ Testing connection to Hugging Face Inference API...")
try:
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say the word 'Test'"}],
        max_tokens=5
    )
    print("✅ SUCCESS! Your token is valid and has Serverless Inference permissions.")
    print(f"🤖 Model Response: {response.choices[0].message.content}")
except Exception as e:
    print("\n❌ API REJECTION. The token loaded, but Hugging Face blocked it.")
    print(f"Reason: {e}")
    print("\n💡 Fix: Go to huggingface.co/settings/tokens, create a 'Fine-grained' token, and ensure 'Make calls to the serverless Inference API' is CHECKED.")
print("========================================")