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
    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig=EvaluationConfig()
    ):
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config and model
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(config.save_directory, f"{type(model).__name__}.pth")))
        repvgg_model_convert(
            self.model,
            save_path=os.path.join(config.save_directory, f"{type(model).__name__}_deploy.pth"),
            do_copy=False
        )
        
        # setup training config
        self.config = config
    
    @torch.no_grad()
    def evaluate(self, dataset: FaceDataset) -> Dict[str, float]:
        # set up dataloaders
        test_loader = data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=8
        )
        self.model.eval()

        # evaluate
        predictions, targets = zip(
            *[
                (
                    self.model(
                        images.to(self.device)
                    ).argmax(dim=-1).cpu(),
                    labels
                )
                for images, labels in tqdm(test_loader, desc="Evaluating")
            ]
        )
        targets = torch.cat(targets).numpy()
        predictions = torch.cat(predictions).numpy()

        # compute metrics
        calculated_metrics = {}
        for metric in self.config.metrics:
            calculated_metrics |= metric.compute(references=targets, predictions=predictions)


        # calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(
            targets,
            predictions,
            normalize="true"
        )
        df_cm = pd.DataFrame(confusion_matrix, index=idx_to_label, columns=idx_to_label)
        print(df_cm)
        
        # plot confusion matrix
        ax = plt.axes()
        sns.heatmap(
            df_cm,
            annot=True,
            ax=ax
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
        EvaluationConfig(),
    ).build()
    args = parser.parse_args()
    config = EvaluationConfig.from_args(args)
    
    # initialise RepVGG model
    model_name = config.model
    create_RepVGG_func = get_RepVGG_func_by_name(model_name)
    model = create_RepVGG_func(deploy=False)

    # classifies 1000 features by default
    # use only 8
    num_classes = len(idx_to_label)
    model.linear.out_features = num_classes
    with torch.no_grad():
        model.linear._parameters["weight"] = model.linear._parameters["weight"][:num_classes,:]
        model.linear._parameters["bias"] = model.linear._parameters["bias"][:num_classes]

    # set up trainer
    evaluator = Evaluator(
        model,
        config=config,
    )

    # evaluate
    test_dataset = FaceDataset("data", split="test", augment=False)
    print(evaluator.evaluate(test_dataset))