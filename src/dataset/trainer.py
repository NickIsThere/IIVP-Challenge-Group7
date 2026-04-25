import torch
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR

class Trainer:
    def __init__(self, model, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
        # Start SWA at 10th epoch
        self.swa_start_epoch = 10
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(loader, desc="Training", leave=False)

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self, loader, current_model=None):
        model_to_eval = current_model if current_model is not None else self.model
        model_to_eval.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model_to_eval(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs):
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)

            if epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                if self.scheduler is not None:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            if epoch >= self.swa_start_epoch:
                torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
                val_loss, val_acc = self.validate_epoch(val_loader, current_model=self.swa_model)
            else:
                val_loss, val_acc = self.validate_epoch(val_loader, current_model=self.model)


            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch + 1:02d}/{epochs} [LR: {current_lr:.6f}] | "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc

        # Save SWA model
        self.model = self.swa_model.module

        return best_acc, self.history