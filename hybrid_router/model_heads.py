import torch
import torch.nn as nn
import numpy as np

class ModelHead(nn.Module):
    def __init__(self, input_dim=768):
        """
        An nn.Module class. Takes input_dim=768 as a parameter.
        
        The architecture is:
            * Linear(768 -> 256)
            * ReLU
            * Dropout(0.2)
            * Linear(256 -> 64)
            * ReLU
            * Linear(64 -> 1)
            * Sigmoid
        Each layer is stored as self.network using nn.Sequential.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Passes x through self.network and returns the result.
        """
        return self.network(x)
    
class ModelHeadCollection:
    def __init__(self, model_names: list[str], device="cpu"):
        """
        Takes a list of model names and a device string (default "cpu").
        In __init__ it builds one ModelHead per model name and stores them in a dict self.heads.
        Calls .to(device) on each head.
        """
        self.heads = {}
        self.device = device

        for model_name in model_names:
            self.heads[model_name] = ModelHead().to(device)
        
    def predict(self, model_name: str, embedding: np.ndarray) -> float:
        """
        Runs one head and returns a plain Python float.
        """
        converted_embedding = torch.tensor(embedding,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.heads[model_name](converted_embedding).item()
    
    def predict_all(self, embedding: np.ndarray) -> dict[str, float]:
        """
        Runs all heads and returns a dict. 
        """
        predictions = {}
        converted_embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        for model_name in self.heads:
            with torch.no_grad():
                predictions[model_name] = self.heads[model_name](converted_embedding).item()

        return predictions