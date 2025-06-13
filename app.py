import gradio as gr
import base64
import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import dotenv_values

env_values = dotenv_values(".env")

# Provider configurations
PROVIDERS = {
    "Chutes": {
        "client": OpenAI(
            base_url="https://llm.chutes.ai/v1/", api_key=env_values["CHUTES_API_KEY"]
        ),
        "models": ["Qwen/Qwen2.5-VL-32B-Instruct"],
    },
    "OpenAI": {
        "client": OpenAI(
            base_url="https://llm.chutes.ai/v1/", api_key=env_values["OPENAI_API_KEY"]
        ),
        "models": ["gpt-4.1", "gpt-4.1-mini"],
    },
    "Groq": {
        "client": OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=env_values["GROQ_API_KEY"],
        ),
        "models": [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ],
    },
    "Gemini": {
        "client": OpenAI(
            api_key=env_values["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "models": [
            "gemini-2.5-flash-preview-05-20",
        ],
    },
    "Ollama": {
        "client": OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama"),
        "models": [
            "gemma3:12b",
            "hf.co/mradermacher/MiniCPM4-8B-GGUF:Q8_0",
            "hf.co/mradermacher/SmolVLM2-2.2B-Instruct-GGUF:F16",
            "gemma3:27b",
        ],
    },
}


SYSTEM_PROMPT = """CONTEXT: You are controlling a motorized Z-axis microscope. Initial reference image shows optimal focus on a similar sample. Subsequent images come from current Z-positions after your repositioning. The Z-axis position is measured by the number of steps from top to bottom in the range of 0 to 4000. The microscope position always starts at 0 i.e., at the very top. THE STEP CAN ONLY BE CHANGED AT LEAST 10 AT A TIME.

ROLE: You are a precision Z-axis control agent responsible for automated focusing that learns from adjustment history. Use human-like trial/error logic with dynamic step sizing to adjust the focus of the image by repositioning the Z-position of the microscope measured in number of steps between the range of 0 to 4000.

ADAPTIVE STRATEGY:
1. Initial coarse search: Large steps (1000-2000 steps) when far from focus
2. Fine-tuning: Reduce step size by 50% when focus improves
3. Overshoot detection: Reverse direction with smaller steps (25%) if focus degrades
4. Convergence: 10 step precision when near optimal focus

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
- Steps: [Number (positive for moving down, negative for moving up, but remember the position cannot be smaller than 0)] (Size: [Coarse/Medium/Fine])
**Rationale:** [Pattern analysis from last 3 positions]
**Expected:** [Predicted focus improvement %]"""


def encode_image_to_base64(image_path):
    """Convert image to base64 for API transmission"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return None


def save_chat_history(
    history, reference_image_path, current_z_position, session_name=None
):
    """Save chat history to JSON file with custom name"""
    try:
        # Create chat_histories folder if it doesn't exist
        histories_dir = "chat_histories"
        os.makedirs(histories_dir, exist_ok=True)

        # Generate filename with custom name or timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name and session_name.strip():
            # Clean the session name for filename
            clean_name = "".join(
                c for c in session_name.strip() if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            clean_name = clean_name.replace(" ", "_")
            filename = f"{clean_name}_{timestamp}.json"
        else:
            filename = f"chat_history_{timestamp}.json"

        filepath = os.path.join(histories_dir, filename)

        # Process history to make it more readable
        processed_history = []
        for i, entry in enumerate(history):
            if isinstance(entry, dict):
                processed_entry = {
                    "message_id": i + 1,
                    "role": entry.get("role", "unknown"),
                    "content": entry.get("content", ""),
                    "timestamp": datetime.now().isoformat(),
                }
                processed_history.append(processed_entry)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                # Handle Gradio's [user_message, bot_message] format
                processed_history.append(
                    {
                        "message_id": f"{i*2 + 1}",
                        "role": "user",
                        "content": str(entry[0]) if entry[0] else "",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                processed_history.append(
                    {
                        "message_id": f"{i*2 + 2}",
                        "role": "assistant",
                        "content": str(entry[1]) if entry[1] else "",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Prepare data to save
        chat_data = {
            "session_metadata": {
                "timestamp": datetime.now().isoformat(),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_messages": len(processed_history),
                "reference_image": (
                    reference_image_path if reference_image_path else None
                ),
                "final_z_position": current_z_position,
                "session_duration": "N/A",
            },
            "chat_history": processed_history,
            "microscope_settings": {
                "z_position_range": "0-4000",
                "step_minimum": 10,
                "current_position": current_z_position,
            },
        }

        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        return f"✅ Chat history saved successfully!\n📁 File: {filename}\n📊 Messages: {len(processed_history)}"

    except Exception as e:
        return f"❌ Error saving chat history: {str(e)}"


def respond(message, history, reference_image, current_z_position, provider, model):
    """Process chat messages and generate microscope control recommendations"""
    current_client = PROVIDERS[provider]["client"]

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
        if model == "gemini-2.5-flash-preview-05-20":
            response = current_client.chat.completions.create(
                model=model,
                reasoning_effort="low",
                messages=messages,
                stream=True,
                max_tokens=800,
                temperature=0.1,
            )
        else:
            response = current_client.chat.completions.create(
                model=model,
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
    gr.Markdown("# 🔬 Microscope Control Assistant")
    gr.Markdown(
        "Upload a reference image showing optimal focus/zoom, set current zoom level, then submit microscope images for adjustment recommendations."
    )

    with gr.Row():
        with gr.Column(scale=1):  # Reference image upload
            reference_image = gr.Image(
                label="Reference Image (Optimal Focus/Zoom)",
                type="filepath",
                height=200,
            )

            # Current z-position level input
            z_input = gr.Number(
                label="Current Z-position",
                value=0,
                minimum=0,
                maximum=4000,
                step=1,
                interactive=True,
                info="Microscope z-position range: 0 - 4000",
            )

            # Quick zoom presets (moved up)
            with gr.Row():
                z_1000 = gr.Button("1000", size="sm")
                z_p1000 = gr.Button("+1000", size="sm")
                z_p500 = gr.Button("+500", size="sm")
                z_p250 = gr.Button("+250", size="sm")
                z_p100 = gr.Button("+100", size="sm")
                z_p50 = gr.Button("+50", size="sm")
                z_p25 = gr.Button("+25", size="sm")
                z_p10 = gr.Button("+10", size="sm")

            # Provider and model selection
            gr.Markdown("### 🤖 AI Provider & Model")
            provider_dropdown = gr.Dropdown(
                choices=list(PROVIDERS.keys()),
                value="Chutes",
                label="Provider",
                interactive=True,
            )

            model_dropdown = gr.Dropdown(
                choices=PROVIDERS["Chutes"]["models"],
                value="Qwen/Qwen2.5-VL-32B-Instruct",
                label="Model",
                interactive=True,
            )

            gr.Markdown("### 💾 Save Session")
            session_name_input = gr.Textbox(
                label="Session Name (Optional)",
                placeholder="Enter a name for this session (e.g., 'Sample_A_Focus_Test')",
                interactive=True,
            )
            save_button = gr.Button("Save Chat History", variant="secondary", size="lg")
            save_status = gr.Textbox(
                label="Save Status",
                placeholder="Chat history save status will appear here...",
                interactive=False,
                lines=2,
            )

            # Instructions
            gr.Markdown(
                """            **Instructions:**
            1. Upload reference image (optimal quality)
            2. Upload current microscope image
            3. Set current z-position
            4. Ask for z-repositioning recommendations
            5. Give your session a name and save chat history when done
            
            **Tips:**
            - Describe what you want to achieve
            - Mention any specific issues you're seeing
            - Use descriptive session names (e.g., 'Sample_A_Focus_Test')
            """
            )

        with gr.Column(scale=2):  # Main chatbot interface using ChatInterface
            chatbot = gr.ChatInterface(
                respond,
                additional_inputs=[
                    reference_image,
                    z_input,
                    provider_dropdown,
                    model_dropdown,
                ],
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

    # Event handlers for quick repositioning buttons
    z_1000.click(lambda: 1000, outputs=z_input)
    z_p1000.click(lambda x: x + 1000, inputs=z_input, outputs=z_input)
    z_p500.click(lambda x: x + 500, inputs=z_input, outputs=z_input)
    z_p250.click(lambda x: x + 250, inputs=z_input, outputs=z_input)
    z_p100.click(lambda x: x + 100, inputs=z_input, outputs=z_input)
    z_p50.click(lambda x: x + 50, inputs=z_input, outputs=z_input)
    z_p25.click(lambda x: x + 25, inputs=z_input, outputs=z_input)
    z_p10.click(
        lambda x: x + 10, inputs=z_input, outputs=z_input
    )  # Event handler for provider dropdown

    def update_models(provider):
        models = PROVIDERS[provider]["models"]
        return gr.Dropdown(choices=models, value=models[0])

    provider_dropdown.change(
        fn=update_models, inputs=[provider_dropdown], outputs=[model_dropdown]
    )
    chat_history_state = gr.State([])

    def handle_save_click(reference_img, z_pos, session_name, history):
        if not history:
            return "⚠️ No chat history to save. Start a conversation first!"
        return save_chat_history(history, reference_img, z_pos, session_name)

    save_button.click(
        fn=handle_save_click,
        inputs=[reference_image, z_input, session_name_input, chatbot.chatbot],
        outputs=save_status,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
