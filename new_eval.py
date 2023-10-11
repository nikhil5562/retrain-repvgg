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
            batch_size = self.config.batch_size,
            shuffle = False,
            num_workers = 8
        )
        self.model.eval()

        predictions, targets = zip(
            *[
                (
                    self.model(
                        images.to(self.device)
                    ).argmax(dim = -1).cpu(),
                    labels
                )
                for images, labels in tqdm(test_loader, desc = "Evaluating")
            ]
        )
        targets = torch.cat(targets).numpy()
        predictions = torch.cat(predictions).numpy()

        # Compute metrics
        calculated_metrics = {}
        for metric in self.config.metrics:
            calculated_metrics |= metric.compute(references = targets, predictions = predictions)

        # Calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(
            targets,
            predictions,
            normalize = "true"
        )
        df_cm = pd.DataFrame(confusion_matrix, index = idx_to_label, columns = idx_to_label)
        print(df_cm)
        
        # Plot confusion matrix
        ax = plt.axes()
        sns.heatmap(
            df_cm,
            annot = True,
            ax = ax
        )
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        plt.show()

        calculated_metrics["normalised_accuracy"] = confusion_matrix[np.eye(len(idx_to_label)).astype(bool)].mean()
        calculated_metrics["confusion_matrix"] = confusion_matrix
        return calculated_metrics

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        EvaluationConfig()
    ).build()
    args = parser.parse_args()
    config = EvaluationConfig.from_args(args)
    
    # Create an Evaluator and perform the evaluation
    model_name = config.model
    create_RepVGG_func = get_RepVGG_func_by_name(model_name)
    model = create_RepVGG_func(deploy = False)
    evaluator = Evaluator(
        model,
        config = config
    )

    # Evaluate
    test_dataset = FaceDataset("data", split = "test", augment = False)
    print(evaluator.evaluate(test_dataset))
