import gradio as gr
import torch
from io import BytesIO
from dotenv import dotenv_values

demo = gr.ChatInterface(
    title="Microscope focus with VLM 📺",
    description="Get suggestions from VLM to focus your microscope",
    textbox=gr.MultimodalTextbox(
        label="Query Input", file_types=["image", ".mp4"], file_count="multiple"
    ),
    stop_btn="Stop Generation",
    multimodal=True,
)
demo.launch(share=True)
