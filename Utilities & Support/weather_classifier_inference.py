import os
import torch
import torch.nn as nn
from torchvision import transforms, models

from PIL import Image


class StudentConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        out = self.pool(f4).flatten(1)
        logits = self.classifier(out)
        return logits, [f1, f2, f3, f4]


class WeatherClassifier:

    def predict_image(self, image_path):
        # -------------------------
        # Config
        # -------------------------
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------
        # Validate Input
        # -------------------------
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist")
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValueError("Only JPG and PNG images are supported")

        # -------------------------
        # Transformations
        # -------------------------
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        # -------------------------
        # Load Model
        # -------------------------
        student = StudentConvNet().to(DEVICE)
        model_path = r"resnet_student.pth"
        state_dict = torch.load(model_path, map_location=DEVICE)
        student.load_state_dict(state_dict)
        student.eval()

        # -------------------------
        # Load and Process Image
        # -------------------------
        image = Image.open(image_path).convert('RGB')  # Load and convert to RGB
        image = transform(image)  # Apply transformations
        image = image.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
        image = image.to(DEVICE)

        # -------------------------
        # Inference
        # -------------------------
        class_names = ['Day', 'Fog', 'Night']
        with torch.no_grad():
            logits, _ = student(image)  # Assuming student returns (logits, features)
            pred = logits.argmax(1).item()  # Get predicted class index
            predicted_class = class_names[pred]

        return predicted_class

#weather_classifier = WeatherClassifier()
#image_path = rf"C:\Users\gg363d\Desktop\Malik\PhD\UCI\Marco Levorato\FUSCO\FUSCO Dataset\DayRainy\EvidenceBasedAnomaly\FallenTree\1339474.png"
#print(weather_classifier.predict_image(image_path=image_path))

# import os

# # Initialize classifier
# weather_classifier = WeatherClassifier()

# # Custom weights per label
# class_weights = {
#     "Day": 1.00,
#     "Fog": 1.30,
#     "Night": 1.25
# }

# # Folder path
# folder_path = r"C:\Users\gg363d\Desktop\Malik\PhD\UCI\Marco Levorato\FUSCO\FUSCO Dataset\DayRainy\EvidenceBasedAnomaly\FallenTree"

# # Score trackers
# total_weighted_score = 0.0
# valid_image_count = 0

# # Loop through each image
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(folder_path, filename)
#         try:
#             label = weather_classifier.predict_image(image_path=image_path).strip()
#             if label not in class_weights:
#                 print(f"{filename} => Unknown class label: '{label}'")
#                 continue

#             weight = class_weights[label]
#             weighted_score = weight  # assuming confidence = 1.0

#             total_weighted_score += weighted_score
#             valid_image_count += 1

#             print(f"{filename} => Class: {label}, Weighted Score: {weighted_score:.2f}")

#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# # Compute average
# if valid_image_count > 0:
#     average_score = total_weighted_score / valid_image_count
#     print(f"\n✅ Average Weighted Score: {average_score:.4f}")
# else:
#     print("❌ No valid images were processed.")