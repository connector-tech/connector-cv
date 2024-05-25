import inspect
from typing import List, Optional, Tuple, Type, Union

import cv2
import numpy as np


def get_providers(device: str) -> List[Tuple[str, Optional[dict]]]:
    """
    Function that defines provider for onnx session.

    Args:
        device (str): [cpu, cuda, cuda:0 ...].

    Returns:
        List[Tuple[str, Optional[dict]]]: provider
    """
    if device == "cpu":
        providers = [("CPUExecutionProvider")]
    elif device == "cuda":
        providers = [("CUDAExecutionProvider")]
    else:
        device_id = int(device.split(":")[1])
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                },
            )
        ]

    return providers


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the sigmoid function value for the input.

    Args:
        x (Union[float, np.ndarray]): Input value or array.

    Returns:
        Union[float, np.ndarray]: Sigmoid function value(s) corresponding
            to the input(s).
    """
    return 1.0 / (1.0 + np.exp(-x))


class HelpMeta(type):
    """
    Metaclass that adds a 'help' method to classes, which prints information
    about methods and their docstrings.

    Methods:
        help(instance, only_call=False)
    """

    def __new__(cls, name, bases, attrs) -> Type:
        """
        Create a new class with the 'help' method.

        Args:
            name (str): Name of the class.
            bases (tuple): Base classes of the new class.
            attrs (dict): Attributes and methods of the new class.

        Returns:
            Type: New class with the 'help' method.
        """
        new_class = super().__new__(cls, name, bases, attrs)

        def help_func(instance, only_call=False):
            """
            Print information about methods and their docstrings in the class.

            Args:
                instance: Instance of the class.
                only_call (bool, optional): If True, only display information about
                    the '__call__' method. Defaults to False.
            """
            for name, func in instance.__class__.__dict__.items():
                if not only_call or name == "__call__":
                    if not (
                            callable(func) and hasattr(func, "__doc__") and func.__doc__
                    ):
                        continue
                    signature = inspect.signature(func)
                    docstring = func.__doc__.replace(" " * 12, "\t").replace(
                        " " * 8, " " * 4
                    )

                    print(f"Function: {name}")
                    print(f"Parameters: {str(signature)}")
                    print(f"Docstring: {docstring}\n")

        new_class.help = help_func
        return new_class


def read_image(image: bytes) -> np.ndarray:
    """
    Read an image from a byte string.

    Args:
        image (bytes): Input image data as a byte string.

    Returns:
        np.ndarray: Image data as a NumPy array.
    """
    nparr = np.frombuffer(image, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
