# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2

# === Label Mapping ===
class_map = {
    "Reach": 0,
    "Pick": 1,
    "Move": 2,
    "Place": 3,
    "Withdraw": 4,
}
reverse_class_map = {v: k for k, v in class_map.items()}

# === Dataset ===
class VideoDataset(Dataset):
    def __init__(self, video_folder, label_folder):
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.video_files = sorted(os.listdir(video_folder))
        self.label_files = sorted(os.listdir(label_folder))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        frames = self.load_video(video_path)

        with open(label_path, "r") as f:
            labels = f.read().splitlines()
            labels = [class_map[label.strip()] for label in labels]

        labels = torch.tensor(labels, dtype=torch.long)

        return frames, labels

    def load_video(self, video_path, target_num_frames=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame.transpose(2, 0, 1)  # HWC -> CHW
            frames.append(torch.tensor(frame, dtype=torch.float32) / 255.0)
            frame_count += 1

        # If the video has fewer frames than required, pad it
        # while frame_count < target_num_frames:
        #     frames.append(torch.zeros(3, 224, 224))  # Padding with zeros

        # # If the video has more frames than required, trim it
        # frames = frames[:target_num_frames]

        cap.release()
        return torch.stack(frames)


# === Model ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(B, T, -1)
        return x

# === Training ===
def train():
    video_folder = "../videos/pick_place"
    label_folder = "../truth/pick_place"

    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4

    dataset = VideoDataset(video_folder, label_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)

            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "saved_model.pth")
    print("Model saved as saved_model.pth")

if __name__ == "__main__":
    train()
