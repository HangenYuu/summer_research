# %%
import os
import cv2

from langchain.agents import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from datetime import datetime

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from dotenv import dotenv_values

from motor_control import StepperMotorController


# Initialization
## API KEY
env_values = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = env_values["OPENAI_API_KEY"]

## Base chatbot
llm = init_chat_model("openai:gpt-4.1")

## Stepper motor controller
controller = StepperMotorController("COM3")
controller.connect()


# %%
## Message State
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The 'next' field indicates where to route to next
    next: str
    sender: str


# %%
@tool
def Change_Position(steps: int) -> None:
    if steps < 0:
        controller.move_motor("right", abs(steps))
    controller.move_motor("left", steps)


@tool
def Capture_Image(camera_index: int = 0, save_path: str | None = None) -> dict:
    """
    Capture an image from the USB microscope (webcam).

    Args:
        camera_index: Camera device index (usually 0 for default camera)
        save_path: Optional path to save the captured image

    Returns:
        dict: Contains status, image data, and file path
    """
    try:
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            return {"status": "error", "message": "Could not open camera"}

        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for manual control

        # Capture frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"status": "error", "message": "Failed to capture image"}

        # Save image if path provided
        if save_path:
            cv2.imwrite(save_path, frame)
        else:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"microscope_image_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)

        return {
            "status": "success",
            "message": "Image captured successfully",
            "image_path": save_path,
            "image_data": frame,
            "shape": frame.shape,
        }

    except Exception as e:
        return {"status": "error", "message": f"Capture failed: {str(e)}"}


# %%
graph_builder = StateGraph(State)
