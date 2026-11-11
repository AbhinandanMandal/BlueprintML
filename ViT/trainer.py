"""Training / evaluation loop."""
import torch

from utils import save_checkpoint, save_experiments


class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device, config, base_dir="./experiments"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config
        self.base_dir = base_dir

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                  f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if (save_model_every_n_epochs > 0
                    and (i + 1) % save_model_every_n_epochs == 0
                    and i + 1 != epochs):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1, base_dir=self.base_dir)
        save_experiments(self.exp_name, self.config, self.model,
                         train_losses, test_losses, accuracies, base_dir=self.base_dir)

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _ = self.model(images)
            predictions = torch.argmax(logits, dim=1)
            total_loss += self.loss_fn(logits, labels).item() * len(images)
            correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss
