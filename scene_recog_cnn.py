import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Scene15Dataset, build_transforms
from model import SceneCNN


def train(train_data_dir, model_dir='trained_cnn.pth', log_file='log.txt', **kwargs):
    """
    Main training procedure for SceneCNN
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== training log begin ===\n")

    def print_and_log(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_and_log(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    train_transform = build_transforms(is_train=True)
    val_transform   = build_transforms(is_train=False)

    # Two separate Dataset instances are created with different transforms.
    # Both are then split with the same random seed so their index subsets
    # are identical, preventing the validation set from receiving augmentation.
    full_train_dataset = Scene15Dataset(data_dir=train_data_dir, transform=train_transform)
    full_val_dataset   = Scene15Dataset(data_dir=train_data_dir, transform=val_transform)

    generator  = torch.Generator().manual_seed(42)
    train_size = 1200
    val_size   = len(full_train_dataset) - train_size

    train_set, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_set   = random_split(full_val_dataset,   [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------ #
    # Model, loss function, and optimiser
    # ------------------------------------------------------------------ #
    model     = SceneCNN(num_classes=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs    = kwargs.get('epochs', 30)
    # Cosine annealing gradually reduces the learning rate, helping the model
    # settle into a sharper minimum in the later stages of training.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Project labels from the 1-15 range to the 0-14 range required
            # by PyTorch's CrossEntropyLoss.
            labels = labels.to(device) - 1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        # --- Validation phase ---
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs  = inputs.to(device)
                labels  = labels.to(device) - 1  # Same 1->0 offset as training.

                outputs   = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc    = 100 * correct / total
        train_loss = running_loss / train_size
        print_and_log(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save only when a new best is reached (best-checkpoint strategy).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir)
            print_and_log(f"--> Saved new best model to {model_dir}")

    print_and_log(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def test(test_data_dir, model_dir='trained_cnn.pth', **kwargs):
    """
    Main inference procedure for SceneCNN
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    test_transform = build_transforms(is_train=False)
    test_dataset   = Scene15Dataset(data_dir=test_data_dir, transform=test_transform)
    test_loader    = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ------------------------------------------------------------------ #
    # Load pre-trained weights
    # ------------------------------------------------------------------ #
    model = SceneCNN(num_classes=15).to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.eval()

    # ------------------------------------------------------------------ #
    # Inference and accuracy calculation
    # ------------------------------------------------------------------ #
    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # Ground-truth labels remain in the 1-15 range.

            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Restore predictions from 0-14 (network output) back to 1-15
            # to match the ground-truth label convention.
            predicted = predicted + 1

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene-15 CNN: training and evaluation.')
    parser.add_argument('--phase',          default='train', choices=['train', 'test'])
    parser.add_argument('--train_data_dir', default='./data/train/',
                        help='Root directory of training images.')
    parser.add_argument('--test_data_dir',  default='./data/test/',
                        help='Root directory of test images.')
    parser.add_argument('--model_dir',      default='trained_cnn.pth',
                        help='Path to save or load the model weights.')
    parser.add_argument('--log_file',       default='./log.txt')
    parser.add_argument('--epochs',         default=30, type=int,
                        help='Number of training epochs.')
    opt = parser.parse_args()

    if opt.phase == 'train':
        train(opt.train_data_dir, opt.model_dir, opt.log_file, epochs=opt.epochs)
    elif opt.phase == 'test':
        test(opt.test_data_dir, opt.model_dir)