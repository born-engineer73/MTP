# predict.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from sklearn.preprocessing import LabelEncoder
from transformers import ViTModel
from tqdm import tqdm

# Config
video_folder = "dataset/videos/"
output_folder = "predicted_labels/"
model_path = "vlm_action_predictor.pth"
num_frames = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Labels
all_labels = ['reach', 'pick', 'move', 'place', 'withdraw', 'pour', 'give']
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Dataset
class VideoDatasetPredict(torch.utils.data.Dataset):
    def __init__(self, video_list, video_folder, transform=None, num_frames=32):
        self.video_list = video_list
        self.video_folder = video_folder
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_list)

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.num_frames)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            if len(frames) == self.num_frames:
                break

        cap.release()
        frames = torch.stack(frames)
        return frames

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        video_path = os.path.join(self.video_folder, f"{video_name}.mp4")
        frames = self.load_video_frames(video_path)
        return frames, video_name

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_videos = list(range(481, 521))  # 481 to 520
test_dataset = VideoDatasetPredict(test_videos, video_folder, transform=transform, num_frames=num_frames)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
class ActionPredictor(nn.Module):
    def __init__(self, num_classes):
        super(ActionPredictor, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.transformer_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=1
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape
        x = x.view(batch_size * frames, C, H, W)
        outputs = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        outputs = outputs.view(batch_size, frames, -1)
        outputs = self.transformer_head(outputs)
        outputs = self.classifier(outputs)
        return outputs

model = ActionPredictor(num_classes=len(label_encoder.classes_)).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Prediction
with torch.no_grad():
    for frames, video_name in tqdm(test_loader):
        frames = frames.to(device)
        outputs = model(frames)
        preds = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
        pred_labels = label_encoder.inverse_transform(preds)

        # Save prediction
        output_path = os.path.join(output_folder, f"{video_name[0]}.txt")
        with open(output_path, "w") as f:
            for label in pred_labels:
                f.write(f"{label}\n")

print("✅ Predictions saved inside 'predicted_labels/' folder")
