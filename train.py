import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# main function
def main():
    DATA_DIR = 'dataset/train'
    MODEL_SAVE_PATH = 'model_art_ai_human.pth'
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder '{DATA_DIR}' tidak ditemukan!")
        print("Buat folder 'dataset/train' dan masukkan subfolder berisi gambar (misal: 'ai', 'human').")
        return
    
    # load data
    print("Loading data...")

    # transform data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    class_names = image_dataset.classes
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Kelas di Temukan: {class_names}")
    print(f"Total Data: {len(image_dataset)}")

    # Setup model
    print("Setting up model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Menggunakan device: {device}")

    # use model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Start Training
    print("Mulai Training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # calculate static
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples = inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

    #  Save model
    print("Saving model...")

    # Save state_dict & class name
    checkpint = {
        'model_state': model.state_dict(),
        'class_names': class_names
    }

    torch.save(checkpint, MODEL_SAVE_PATH)

    print("Model saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()