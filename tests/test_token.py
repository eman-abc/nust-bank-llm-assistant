import os

import pytest
from huggingface_hub import InferenceClient


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN is not configured for integration testing.")
def test_hf_token_can_call_inference_api():
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=os.getenv("HF_TOKEN"))
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say the word Test"}],
        max_tokens=5,
    )

    assert response.choices[0].message.content
