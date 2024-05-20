import torch
import torch.nn as nn
import albumentations as A
import torchvision.models as models


class LivenessPytorch:
    def __init__(self, model_file, device="cpu", threshold=0.5):
        """_summary_

        Args:
            model_file (str): Path to the model checkpoint file.
            device (str, optional): Device to load the model onto("cpu", "cuda").
                Defaults to "cpu".
            threshold (float, optional): Threshold. Defaults to 0.5.
        """
        self.model = models.efficientnet_b3()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        self.model.load_state_dict(torch.load(model_file))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.imsize = 300
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = A.Compose(
            [
                A.Resize(self.imsize, self.imsize, p=1),
                A.Normalize(mean=self.mean, std=self.std, p=1),
            ]
        )

    @torch.no_grad()
    def __call__(self, image):
        """
        Predicts whether an image is live or fake using the model's __call__ method.

        Args:
            image (np.ndarray): Input image as a NumPy array (H, W, C).

        Returns:
            bool: True if the image is predicted to be live, False otherwise.
        """
        image = self.transform(image=image)["image"]

        image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()
        image = image.to(self.device)
        output = self.model(image)

        output = torch.sigmoid(output).item()

        # output = self.model(image)
        return output
