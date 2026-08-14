"""Smoke test: trace a single image to LangSmith via ``wrap_openai``.

Verifies that a multimodal image is rendered in the LangSmith trace UI.

Run:
    python -m scripts.test_single_image_trace

Then check the printed LangSmith URL in your browser.
"""

from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith import Client as LangSmithClient
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from config.settings import ENHANCEMENT_BASE_URL, ENHANCEMENT_MODEL, HF_TOKEN
from src.logger import get_logger

logger = get_logger(__name__)

IMAGE_PATH = "dbv2/images/AN699chunks_c1_img0.jpg"
MIME_TYPE = "image/jpeg"


def main() -> None:
    """Send a single base64 image to the vision model and print the trace URL."""
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    client = wrap_openai(
        OpenAI(api_key=HF_TOKEN, base_url=ENHANCEMENT_BASE_URL),
        chat_name="SingleImageTrace",
    )

    response = client.chat.completions.create(
        model=ENHANCEMENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in a few words."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{MIME_TYPE};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    logger.info("Answer: %s", response.choices[0].message.content)

    ls = LangSmithClient()
    runs = ls.list_runs(project_name="Multimodal Rag", run_type="llm", limit=1)
    for run in runs:
        logger.info("LangSmith run id: %s", run.id)
        logger.info(
            "Trace URL: https://smith.langchain.com/o/1/projects/p/%s?display=long",
            run.id,
        )


if __name__ == "__main__":
    main()
