import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Dict

from model import DigitRecognizer


def load_model(
        model_architecture: nn.Module,
        model_path: str = "best_model.pth", 
        device: torch.device = torch.device("cuda")
    ) -> DigitRecognizer:
    """
    Loads the model from a .pth file for inference.
    
    args:
    - model_architecture (nn.Module): an instance of the class in which the model architecture is defined
    - model_path (str): the path to the .pth model file
    - device (torch.device): specifies device to use model on

    returns:
    - the loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_architecture
    
    try:
        checkpoint = torch.load(model_path, map_location=device)

        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"An error occurred when loading the model: {e}")
        return None
    
    model.eval()
    model.to(device)
    
    print(f"Model loaded to: {device}")
    return model


def single_image_inference(
        model_path: str, 
        image_array: np.ndarray, 
        model_architecture: nn.Module
    ) -> Dict:
    """
    Loads the trained model and performs prediction with the model on an image.

    args:
    - model_path (str): the path to the .pth model file
    - image_array (np.ndarray): the image in the form or an array
    - model_architecture (nn.Module): an instance of the class in which the model architecture is defined

    returns:
    - a Python dict containing the predicted class (digit) and its corresponding probability
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(
        model_architecture=model_architecture,
        model_path=model_path,
        device=device
    )

    if model is None:
        return {
            "predicted_class": "",
            "probability": ""
        }
    
    if len(image_array.shape) == 2:
        image_array = image_array.reshape(1, 1, 28, 28)             # (batch_size, num_channels, H, W) = (1, 1, 28, 28)

    image_tensor_torch = torch.from_numpy(image_array).float()
    image_tensor = image_tensor_torch / 255.0                       # normalize to [0.0, 1.0]
    transform = transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
    image_tensor = transform(image_tensor)                          # transform to [-1.0, 1.0] 
    image_tensor = image_tensor.to(device)
    
    predictions = model(image_tensor)
    probabilities = torch.softmax(predictions, dim=1)

    predicted_class = torch.argmax(probabilities)
    probability = torch.max(probabilities)

    prediction = {
        "predicted_class": predicted_class,
        "probability": probability
    }

    print(f"Predicted result: {prediction['predicted_class']}")
    print(f"Probability: {prediction["probability"]}")

    return prediction


def main():
    """
    The main function to run.
    """
    rng = np.random.default_rng(seed=34567)

    # randomly generate an array for testing
    image = rng.integers(low=0, high=255, size=(28,28))

    model_path = "best_model.pth"

    model_architecture = DigitRecognizer()

    predictions = single_image_inference(
        model_path=model_path,
        image_array=image,
        model_architecture=model_architecture
    )


if __name__ == "__main__":
    main()
