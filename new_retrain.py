import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
from dataset import FaceDataset, idx_to_label
from ml_utils import Config, ParserBuilder
from tqdm import tqdm
from repvgg import create_RepVGG_A0, repvgg_model_convert
import wandb
from dataclasses import asdict, dataclass
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "repvgg"))
from repvgg import get_RepVGG_func_by_name

@dataclass
class TrainingConfig(Config):
    model: str = "RepVGG-A0"
    epochs: int = 50
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    save_directory: str = "checkpoints"
    wandb_project: str = "emotion"
    training_patience: int = 5
    freeze_layer: str = "rbr_1x1"  # Change this to specify the layer to freeze

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig=TrainingConfig()
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.save_directory = config.save_directory
        self.config = config

        # Define the class weights on the same device as the model
        class_weights = {
            'anger': 0.9159,
            'contempt': 19.4716,
            'disgust': 5.8882,
            'fear': 0.9148,
            'happy': 0.4633,
            'neutral': 0.7665,
            'sad': 0.7607,
            'surprise': 1.2251
        }

        # Move the weight tensor to the same device as the model
        self.class_weights = torch.Tensor(list(class_weights.values())).to(self.device)

        # Define the loss function (criterion) with class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        if self.config.freeze_layer == "rbr_dense":
            self.freeze_layers(self.model.stage1)
            print("Freezing layers in rbr_dense")
        elif self.config.freeze_layer == "rbr_1x1":
            self.freeze_layers(self.model.stage2)
            print("Freezing layers in rbr_1x1")

    def freeze_layers(self, stage):
        for param in stage.parameters():
            param.requires_grad = False

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)  # Use the defined criterion
        return loss

    def train(self, train_dataset: FaceDataset, val_dataset: FaceDataset):
        wandb.init(project=self.config.wandb_project, config=asdict(self.config), dir=self.save_directory)
        self.model.to(self.device)

        optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad], lr=self.config.learning_rate, weight_decay=5e-2)
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
            self.model.train()
            train_loss = 0.
            t = tqdm(train_loader, desc=f"Training ({epoch+1:02})", total=len(train_loader))
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
            for i, batch in enumerate(t):
                if i % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                losses = self._compute_loss(batch)
                loss = losses.detach().cpu().item()
                train_loss += loss
                t.set_postfix(loss=loss)
                losses.backward()
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
            train_loss /= len(train_loader)
            metrics["loss"] = train_loss
            self.model.eval()
            val_loss = 0.
            t = tqdm(val_loader, desc=f"Evaluating ({epoch+1:02}", total=len(val_loader))
            with torch.no_grad():
                for batch in t:
                    losses = self._compute_loss(batch)
                    loss = losses.detach().cpu().item()
                    val_loss += loss
                    t.set_postfix(loss=loss)
            val_loss /= len(val_loader)
            metrics["val_loss"] = val_loss
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = self.config.training_patience
                torch.save(self.model.state_dict(), os.path.join(self.save_directory, f"{type(self.model).__name__}.pth"))
            else:
                patience -= 1
                if patience == 0:
                    break
            wandb.log(metrics)
        deployable_model = repvgg_model_convert(self.model, save_path=None, do_copy=True)
        torch.save(deployable_model.state_dict(), "deployable_model.pth")

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(TrainingConfig()).build()
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)

    model_name = config.model
    create_RepVGG_func = get_RepVGG_func_by_name(model_name)
    model = create_RepVGG_func(deploy=False)

    state_dict = torch.load(f"pretrained/{model_name}-train.pth")
    num_classes = len(idx_to_label)
    model.linear.out_features = num_classes
    with torch.no_grad():
        state_dict["linear.weight"] = state_dict["linear.weight"][:len(idx_to_label),:]
        state_dict["linear.bias"] = state_dict["linear.bias"][:len(idx_to_label)]

    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer '{name}' is frozen.")

    trainer = Trainer(model, config=config)
    train_dataset = FaceDataset("data", split="train", augment=True)
    val_dataset = FaceDataset("data", split="val", augment=False)
    trainer.train(train_dataset, val_dataset)