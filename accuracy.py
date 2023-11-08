import torch.utils.data as data
import torch.nn as nn
import numpy as np
import torch

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import evaluate

from dataset import FaceDataset, idx_to_label
from tqdm import tqdm

from ml_utils import Config, ParserBuilder
from dataclasses import dataclass, field
from typing import Dict, List
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "repvgg"))
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

@dataclass
class EvaluationConfig(Config):
    model: str = "RepVGG-A0"
    """RepVGG model to use"""
    batch_size: int = 64
    """Batch size to use for evaluating."""
    save_directory: str = "checkpoints"
    """Directory to load checkpoints from."""
    additional_metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    """Additional metrics to compute."""
    @property
    def metrics(self):
        return [
            evaluate.load(metric)
            for metric in self.additional_metrics
        ]

class Evaluator:
    def __init__(self, model: nn.Module, config: EvaluationConfig = EvaluationConfig()):
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model with the correct architecture
        model_name = config.model
        create_RepVGG_func = get_RepVGG_func_by_name(model_name)
        self.model = create_RepVGG_func(deploy=False)

        # Load pretrained weights
        state_dict = torch.load(f"pretrained/{model_name}-train.pth")
        num_classes = len(idx_to_label)
        self.model.linear.out_features = num_classes
        with torch.no_grad():
            state_dict["linear.weight"] = state_dict["linear.weight"][:len(idx_to_label), :]
            state_dict["linear.bias"] = state_dict["linear.bias"][:len(idx_to_label)]
        self.model.load_state_dict(state_dict, strict=False)

        # Setup the save directory
        self.save_directory = config.save_directory
    
        # Setup training config
        self.config = config

        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def evaluate(self, dataset: FaceDataset):
        # Set up dataloaders
        test_loader = data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=8
        )
        self.model.eval()
        self.model.to(self.device)

        correct_predictions = 0
        total_predictions = 0

        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            predictions = outputs.argmax(dim=-1)

            # Check if the predictions are correct and update the counts
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += len(labels)

        # Calculate accuracy as a percentage
        accuracy = (correct_predictions / total_predictions) * 100

        # Print and return accuracy
        print(f"Accuracy: {accuracy:.2f}%")
        return {"accuracy": accuracy}

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        EvaluationConfig()
    ).build()
    args = parser.parse_args()
    config = EvaluationConfig.from_args(args)

    # Create an Evaluator and perform the evaluation
    model_name = config.model
    create_RepVGG_func = get_RepVGG_func_by_name(model_name)
    model = create_RepVGG_func(deploy=False)
    evaluator = Evaluator(
        model,
        config=config
    )

    # Evaluate
    test_dataset = FaceDataset("data", split="test", augment=False)
    print(evaluator.evaluate(test_dataset))
