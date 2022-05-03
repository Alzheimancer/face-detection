import os

import numpy as np
import torch

from .alignment import load_net, batch_detect


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


def get_image(img_path):
    if type(img_path) == str:  # Load from file path
        if not os.path.isfile(img_path):
            raise ValueError(
                "Input image file path (", img_path, ") does not exist.")
        img = cv2.imread(img_path)

    elif isinstance(img_path, np.ndarray):  # Use given NumPy array
        img = img_path.copy()

    else:
        raise ValueError(
            "Invalid image input. Only file paths or a NumPy array accepted.")

    # Validate image shape
    if len(img.shape) != 3 or np.prod(img.shape) == 0:
        raise ValueError(
            "Input image needs to have 3 channels at must not be empty.")

    return img


class RetinaFace:
    def __init__(
        self,
        gpu_id=-1,
        model_path=relative("weights/mobilenet0.25_Final.pth"),
        network="mobilenet",
    ):
        self.gpu_id = gpu_id
        self.device = (
            torch.device("cpu") if gpu_id == -
            1 else torch.device("cuda", gpu_id)
        )
        self.model = load_net(model_path, self.device, network)

    def detect(self, img_path):

        images = get_image(img_path)

        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return batch_detect(self.model, [images], self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), self.device)
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        else:
            raise NotImplementedError()

    def __call__(self, images):
        return self.detect(images)
