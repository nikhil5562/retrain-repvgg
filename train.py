import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch

from dataset import FaceDataset, idx_to_label
from ml_utils import Config, ParserBuilder
from tqdm import tqdm
import wandb

from dataclasses import asdict, dataclass
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "repvgg"))
from repvgg import get_RepVGG_func_by_name

@dataclass
class TrainingConfig(Config):
    model: str = "RepVGG-A0"
    """RepVGG model to use"""
    epochs: int = 50
    """Number of epochs to train for."""
    batch_size: int = 64
    """Batch size to use for training."""
    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps to use."""
    learning_rate: float = 1e-4
    """Learning rate to use for training."""
    save_directory: str = "checkpoints"
    """Directory to save checkpoints to."""
    wandb_project: str = "emotion"
    """Wandb project to log to."""
    training_patience: int = 5


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig=TrainingConfig()
    ):
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config and model
        self.model = model.to(self.device)
        self.save_directory = config.save_directory
    
        # setup training config
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

    def train(self, train_dataset: FaceDataset, val_dataset: FaceDataset):
        wandb.init(project=self.config.wandb_project, config=asdict(self.config), dir=self.save_directory)

        self.model.to(self.device)
        optimizer = optim.Adam([param for param in self.model.parameters() if param.is_leaf], lr=self.config.learning_rate, weight_decay=5e-2)

        # set up dataloaders
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=8
        )

        patience = self.config.training_patience
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            metrics = {}
            # train
            self.model.train()
            train_loss = 0.
            t = tqdm(train_loader, desc=f"Training ({epoch+1:02})", total=len(train_loader))
            gradient_acculumation_steps = self.config.gradient_accumulation_steps
            for i, batch in enumerate(t):
                if i % gradient_acculumation_steps == 0:
                    optimizer.zero_grad()

                losses = self._compute_loss(batch)
                loss = losses.detach().cpu().item()
                train_loss += loss

                t.set_postfix(loss=loss)
                losses.backward()
                if (i + 1) % gradient_acculumation_steps == 0:
                    optimizer.step()
            train_loss /= len(train_loader)
            metrics["loss"] = train_loss

            # validate
            self.model.eval()
            val_loss = 0.
            t = tqdm(val_loader, desc=f"Evaluating ({epoch+1:02})", total=len(val_loader))
            with torch.no_grad():
                for batch in t:
                    losses = self._compute_loss(batch)
                    loss = losses.detach().cpu().item()
                    val_loss += loss

                    t.set_postfix(loss=loss)
            val_loss /= len(val_loader)
            metrics["val_loss"] = val_loss
            
            # early stopping and save model
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = self.config.training_patience
                torch.save(self.model.state_dict(), os.path.join(self.save_directory, f"{type(model).__name__}.pth"))
            else:
                patience -= 1
                if patience == 0:
                    break
            # save model
            wandb.log(metrics)


if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        TrainingConfig()
    ).build()
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)

    # initialise RepVGG model
    model_name = config.model
    create_RepVGG_func = get_RepVGG_func_by_name(model_name)
    model = create_RepVGG_func(deploy=False)

    # load pretrained model
    # pretrained weights are available at https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq and saved in pretrained/
    model.load_state_dict(torch.load(f"pretrained/{model_name}-train.pth"))

    # classifies 1000 features by default
    # use only 8
    num_classes = len(idx_to_label)
    model.linear.out_features = num_classes
    with torch.no_grad():
        model.linear._parameters["weight"] = model.linear._parameters["weight"][:num_classes,:]
        model.linear._parameters["bias"] = model.linear._parameters["bias"][:num_classes]

    # set up trainer
    trainer = Trainer(
        model,
        config=config,
    )

    # train
    train_dataset = FaceDataset("data", split="train", augment=True)
    val_dataset = FaceDataset("data", split="val", augment=False)
    trainer.train(train_dataset, val_dataset)