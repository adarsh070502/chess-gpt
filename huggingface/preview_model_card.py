"""
Preview the model card (model_repo/README.md) locally, before uploading.
Hugging Face hides the YAML header (between the first two '---' lines) and shows its fields as tags; same here.

Run:  python chess_gpt/huggingface/preview_model_card.py   -> http://127.0.0.1:7864
"""
from pathlib import Path

import gradio as gr

text = (Path(__file__).parent / "model_repo" / "README.md").read_text(encoding="utf-8")
_, header, body = text.split("---\n", 2)     # "", YAML header, card

with gr.Blocks(title="Model card preview") as demo:
    with gr.Accordion("YAML header (shown as tags / metadata on Hugging Face)", open=False):
        gr.Code(header, language="yaml")
    gr.Markdown(body)

if __name__ == "__main__":
    demo.launch(server_port=7864, inbrowser=True)
