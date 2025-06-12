import serial
import time


class StepperMotorController:
    def __init__(self, port="COM3", baudrate=9600):
        """
        Initialize the stepper motor controller

        Args:
            port (str): Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate (int): Communication speed (must match Arduino)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None

    def connect(self):
        """Establish connection to Arduino"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Close the serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Disconnected from Arduino")

    def move_motor(self, direction, steps):
        """
        Function to control stepper motor - can be called by LLM agent

        Args:
            direction (str): 'left' or 'right'
            steps (int): Number of steps to move

        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("Error: Not connected to Arduino")
            return False

        try:
            if direction.lower() == "left":
                command = f"L{steps}"
            elif direction.lower() == "right":
                command = f"R{steps}"
            else:
                print("Error: Direction must be 'left' or 'right'")
                return False

            # Send command
            self.serial_connection.write(command.encode("utf-8"))

            # Wait for and read response
            time.sleep(0.1)
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode("utf-8").strip()
                print(f"Arduino response: {response}")

            return True

        except Exception as e:
            print(f"Error sending command: {e}")
            return False

    def stop_motor(self):
        """Stop the motor"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(b"S")
            time.sleep(0.1)
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode("utf-8").strip()
                print(f"Arduino response: {response}")
            return True
        return False


# Function that can be called by LLM agent
def control_stepper_motor(direction, steps, port="COM3"):
    """
    Simple function for LLM agent to control stepper motor

    Args:
        direction (str): 'left' or 'right'
        steps (int): Number of steps to move
        port (str): Serial port of Arduino

    Returns:
        bool: Success status
    """
    controller = StepperMotorController(port)

    if controller.connect():
        success = controller.move_motor(direction, steps)
        time.sleep(1)  # Allow movement to complete
        controller.disconnect()
        return success
    else:
        return False
