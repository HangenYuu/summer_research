import gradio as gr
import base64
import time
from openai import OpenAI
from dotenv import dotenv_values

env_values = dotenv_values(".env")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=env_values["GROQ_API_KEY"],
)

# Enhanced system prompt following best practices
SYSTEM_PROMPT = """You are an expert microscopy assistant specialized in zoom and focus optimization.

ROLE: Analyze microscope images and provide precise zoom/focus adjustment instructions.

MICROSCOPE SPECIFICATIONS:
- Zoom range: 50x to 1000x (strictly enforced)
- Focus control: Coarse and fine adjustment available
- Current zoom level will be provided with each image

REFERENCE STANDARD:
You have access to a reference image showing optimal focus and zoom quality. Use this as your benchmark for all comparisons.

ANALYSIS REQUIREMENTS:
1. Compare current image quality to the reference image
2. Assess sharpness, contrast, detail clarity, and appropriate magnification
3. Consider current zoom level when recommending changes
4. Provide specific, actionable instructions

RESPONSE FORMAT (always include all sections):
**Focus Assessment:** [Poor/Fair/Good/Excellent]
**Current Zoom Evaluation:** [Too Low/Appropriate/Too High]
**Recommended Actions:**
- Zoom adjustment: [specific level or direction]
- Focus adjustment: [coarse/fine, direction]
**Reasoning:** [brief technical justification]
**Expected Outcome:** [what user should see after adjustments]

GUIDELINES:
- Be concise and precise
- Use bullet points for clarity
- Focus on practical, actionable advice
- Recommend zoom levels within 50x-1000x range only
- Prioritize image quality over magnification level"""


def encode_image_to_base64(image_path):
    """Convert image to base64 for API transmission"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return None


def respond(message, history, reference_image, current_zoom):
    """Process chat messages and generate microscope control recommendations"""

    # Prepare messages for the API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add reference image context if available
    if reference_image is not None:
        reference_base64 = encode_image_to_base64(reference_image)
        if reference_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "REFERENCE IMAGE - This shows optimal focus and zoom quality. Use this as your standard for comparison:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{reference_base64}"
                            },
                        },
                    ],
                }
            )

    # Process chat history - corrected for messages format
    for entry in history:
        if entry["role"] == "user":
            if isinstance(entry["content"], tuple):
                # This is a file upload from history
                file_path = entry["content"][0]
                file_base64 = encode_image_to_base64(file_path)
                if file_base64:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{file_base64}"
                                    },
                                },
                            ],
                        }
                    )
            else:
                # This is text content
                messages.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            messages.append({"role": "assistant", "content": entry["content"]})

    # Handle current message with image
    if message["files"]:
        current_image_path = message["files"][0]
        current_base64 = encode_image_to_base64(current_image_path)

        if current_base64:
            context_message = f"""CURRENT MICROSCOPE STATUS:
- Zoom Level: {current_zoom}x
- Request: {message['text'] if message['text'] else 'Please analyze this image and provide zoom/focus recommendations'}

CURRENT IMAGE TO ANALYZE:"""

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{current_base64}"
                            },
                        },
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Current zoom: {current_zoom}x. {message['text']}",
                }
            )
    else:
        # Text-only message
        messages.append(
            {
                "role": "user",
                "content": f"Current zoom level: {current_zoom}x. {message['text']}",
            }
        )

    try:
        # Call the API with streaming
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            stream=True,
            max_tokens=800,
            temperature=0.1,  # Lower temperature for more consistent technical advice
        )

        # Stream the response
        partial_message = ""
        for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    partial_message += delta.content
                    yield partial_message

    except Exception as e:
        yield f"Error processing request: {str(e)}"


# Create the Gradio interface using modern ChatInterface
with gr.Blocks(
    title="Microscope Zoom Control Assistant", theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# 🔬 Microscope Zoom Control Assistant")
    gr.Markdown(
        "Upload a reference image showing optimal focus/zoom, set current zoom level, then submit microscope images for adjustment recommendations."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Reference image upload
            reference_image = gr.Image(
                label="Reference Image (Optimal Focus/Zoom)",
                type="filepath",
                height=200,
            )

            # Current zoom level input
            zoom_input = gr.Number(
                label="Current Zoom Level",
                value=50,
                minimum=50,
                maximum=1000,
                step=10,
                interactive=True,
                info="Microscope zoom range: 50x - 1000x",
            )

            # Quick zoom presets
            with gr.Row():
                zoom_50 = gr.Button("50x", size="sm")
                zoom_100 = gr.Button("100x", size="sm")
                zoom_200 = gr.Button("200x", size="sm")
                zoom_400 = gr.Button("400x", size="sm")
                zoom_1000 = gr.Button("1000x", size="sm")

            # Instructions
            gr.Markdown(
                """
            **Instructions:**
            1. Upload reference image (optimal quality)
            2. Set current zoom level
            3. Upload current microscope image
            4. Ask for zoom/focus recommendations
            
            **Tips:**
            - Include current zoom level in your message
            - Describe what you want to achieve
            - Mention any specific issues you're seeing
            """
            )

        with gr.Column(scale=2):
            # Main chatbot interface using ChatInterface
            chatbot = gr.ChatInterface(
                respond,
                additional_inputs=[reference_image, zoom_input],
                title=None,
                description=None,
                multimodal=True,
                textbox=gr.MultimodalTextbox(
                    placeholder="Upload current microscope image and describe what you need help with...",
                    file_count="single",
                    file_types=["image"],
                ),
                type="messages",
            )

    # Event handlers for quick zoom buttons
    zoom_50.click(lambda: 50, outputs=zoom_input)
    zoom_100.click(lambda: 100, outputs=zoom_input)
    zoom_200.click(lambda: 200, outputs=zoom_input)
    zoom_400.click(lambda: 400, outputs=zoom_input)
    zoom_1000.click(lambda: 1000, outputs=zoom_input)

if __name__ == "__main__":
    demo.launch(debug=True)
