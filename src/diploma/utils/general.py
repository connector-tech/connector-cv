import inspect
from typing import List, Optional, Tuple, Type, Union

import torch
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


def to_numpy(array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert input array data to a NumPy array.

    Args:
        array (Union[torch.Tensor, np.ndarray]): Input array data as a tensor or
            Numpy array.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    elif isinstance(array, list):
        return np.array(array)
    else:
        return array


def minmax(
    img: Union[torch.Tensor, np.ndarray], asint: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the input image to have values in the range [0, 1] or [0, 255].

    Args:
        img (Union[torch.Tensor, np.ndarray]): Input image data as a tensor or np.ndarray.
        asint (bool, optional): If True, the output image values will be
            scaled to [0, 255] and converted to integers. Defaults to False.

    Returns:
        Union[torch.Tensor, np.ndarray]: Normalized image data.
    """
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    if asint:
        img = img * 255
        if type(img) is torch.Tensor:
            img = img.long()
        else:
            img = img.astype("uint8")

    return img


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
