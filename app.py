import gradio as gr
import base64
from openai import OpenAI
from dotenv import dotenv_values

env_values = dotenv_values(".env")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=env_values["GROQ_API_KEY"],
)

SYSTEM_PROMPT = """CONTEXT: You are controlling a motorized Z-axis microscope. Initial reference image shows optimal focus on a similar sample. Subsequent images come from current Z-positions after your repositioning. The Z-axis position is measured by the number of steps from top to bottom in the range of 0 to .... The microscope position always starts at 0.

ROLE: You are a precision Z-axis control agent responsible for automated focusing that learns from adjustment history. Use human-like trial/error logic with dynamic step sizing to adjust the focus of the image by repositioning the Z-position of the microscope measured in number of steps between the range of 0 to ....

ADAPTIVE STRATEGY:
1. Initial coarse search: Large steps (±10-20 steps) when far from focus
2. Fine-tuning: Reduce step size by 50% when focus improves
3. Overshoot detection: Reverse direction with smaller steps (25%) if focus degrades
4. Convergence: <MIN_STEP> step precision when near optimal focus

ANALYSIS PROCESS:
1. Compare current image to reference image to determine if focus is achieved
2. Track focus quality trajectory by comparing to previous images after adjustments
3. Calculate optimal next step size/direction using historical data
mas
RESPONSE FORMAT:
**Focus State:** 
- Quality: [% match to reference] 
- Trend: [Improving/Declining/Stagnant]
**Adjustment:** 
- Direction: [Up/Down] 
- Steps: [Number] (Size: [Coarse/Medium/Fine])
**Rationale:** [Pattern analysis from last 3 positions]
**Expected:** [Predicted focus improvement %]

NEVER INCLUDE: Technical specs, safety disclaimers, or hardware details."""


def encode_image_to_base64(image_path):
    """Convert image to base64 for API transmission"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return None


def respond(message, history, reference_image, current_z_position):
    """Process chat messages and generate microscope control recommendations"""
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
                            "text": "REFERENCE IMAGE - This shows optimal focus quality for a similar sample. Use this as your standard for comparison:",
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
- Z-position: {current_z_position}
- Request: {message['text'] if message['text'] else 'Please analyze this image together with the adjustment history results and provide repositioning recommendations'}

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
                    "content": f"Current Z-position: {current_z_position}. {message['text']}",
                }
            )
    else:
        # Text-only message
        messages.append(
            {
                "role": "user",
                "content": f"Current Z-position level: {current_z_position}. {message['text']}",
            }
        )

    try:
        # Call the API with streaming
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            stream=True,
            max_tokens=800,
            temperature=0.1,
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

            # Current z-position level input
            z_input = gr.Number(
                label="Current Z-position Level",
                value=50,
                minimum=0,
                maximum=1000,
                step=1,
                interactive=True,
                info="Microscope z-position range: 0 - ",
            )

            # Quick zoom presets
            with gr.Row():
                z_15 = gr.Button("15", size="sm")
                # zoom_100 = gr.Button("100x", size="sm")
                # zoom_200 = gr.Button("200x", size="sm")
                # zoom_400 = gr.Button("400x", size="sm")
                # zoom_1000 = gr.Button("1000x", size="sm")

            # Instructions
            gr.Markdown(
                """
            **Instructions:**
            1. Upload reference image (optimal quality)
            2. Upload current microscope image
            3. Set current z-position
            4. Ask for z-repositioning recommendations
            
            **Tips:**
            - Describe what you want to achieve
            - Mention any specific issues you're seeing
            """
            )

        with gr.Column(scale=2):
            # Main chatbot interface using ChatInterface
            chatbot = gr.ChatInterface(
                respond,
                additional_inputs=[reference_image, z_input],
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
    z_15.click(lambda: 15, outputs=z_input)
    # zoom_100.click(lambda: 100, outputs=z_input)
    # zoom_200.click(lambda: 200, outputs=z_input)
    # zoom_400.click(lambda: 400, outputs=z_input)
    # zoom_1000.click(lambda: 1000, outputs=z_input)

if __name__ == "__main__":
    demo.launch(debug=True)
