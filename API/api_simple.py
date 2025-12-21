from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import io
import uvicorn

MODEL_PATH = "best_yolo_13_30.pth"
IMG_SIZE = 416
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: "Spaces",
    1: "Free",
    2: "Occupied"
}

CLASS_COLORS = {
    0: 'red',
    1: 'lime',
    2: 'blue'
}

class CraterDetectorResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        self.refine = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.detect = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 5 + num_classes, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.refine(x)
        x = self.detect(x)
        return x

def load_model():
    model = CraterDetectorResNet(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def detect_parking_spaces(image: Image.Image, model: nn.Module, conf_threshold: float = 0.3) -> list:
    original_size = image.size

    image_resized = _resize_with_padding(image, IMG_SIZE)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = transforms.ToTensor()(image_resized)
    img_tensor = normalize(img_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    boxes = _decode_predictions(pred, conf_threshold)
    boxes = _apply_nms(boxes, iou_threshold=0.2)
    boxes = _scale_boxes(boxes, original_size, IMG_SIZE)

    return boxes


def _resize_with_padding(image: Image.Image, target_size: int) -> Image.Image:
    image = image.convert('RGB')
    old_size = image.size

    ratio = float(target_size) / max(old_size)
    new_size = tuple(int(x * ratio) for x in old_size)

    image = image.resize(new_size, Image.BILINEAR)
    new_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    new_image.paste(image, ((target_size - new_size[0]) // 2,
                            (target_size - new_size[1]) // 2))

    return new_image


def _decode_predictions(pred: torch.Tensor, conf_threshold: float = 0.3) -> list:
    if pred.dim() == 4:
        pred = pred.squeeze(0)

    C, S, _ = pred.shape
    cell_size = IMG_SIZE / S
    boxes = []

    xy = torch.sigmoid(pred[0:2])
    wh = torch.sigmoid(pred[2:4])
    obj_conf = torch.sigmoid(pred[4:5])
    class_probs = torch.softmax(pred[5:], dim=0)

    for gy in range(S):
        for gx in range(S):
            conf = obj_conf[0, gy, gx].item()
            class_prob = class_probs[:, gy, gx]
            class_id = torch.argmax(class_prob).item()
            score = conf * class_prob[class_id].item()

            if score < conf_threshold:
                continue

            cx = (gx + xy[0, gy, gx].item()) * cell_size
            cy = (gy + xy[1, gy, gx].item()) * cell_size
            w = wh[0, gy, gx].item() * IMG_SIZE
            h = wh[1, gy, gx].item() * IMG_SIZE

            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(IMG_SIZE, cx + w / 2)
            y2 = min(IMG_SIZE, cy + h / 2)

            if x1 >= x2 or y1 >= y2:
                continue

            boxes.append((x1, y1, x2, y2, score, class_id))

    return boxes

def _apply_nms(boxes: list, iou_threshold: float = 0.3) -> list:
    if not boxes:
        return []

    import torchvision.ops
    boxes_tensor = torch.tensor([b[:4] for b in boxes], dtype=torch.float32)
    scores_tensor = torch.tensor([b[4] for b in boxes])
    labels_tensor = torch.tensor([b[5] for b in boxes], dtype=torch.int64)

    keep = torchvision.ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, iou_threshold)

    return [boxes[i] for i in keep.tolist()]


def _scale_boxes(boxes: list, original_size: tuple, model_input_size: int) -> list:
    orig_w, orig_h = original_size
    ratio = model_input_size / max(orig_w, orig_h)

    new_size_w = int(orig_w * ratio)
    new_size_h = int(orig_h * ratio)

    pad_x = (model_input_size - new_size_w) // 2
    pad_y = (model_input_size - new_size_h) // 2

    scaled_boxes = []
    for x1, y1, x2, y2, score, class_id in boxes:
        x1_unpadded = (x1 - pad_x) / ratio
        y1_unpadded = (y1 - pad_y) / ratio
        x2_unpadded = (x2 - pad_x) / ratio
        y2_unpadded = (y2 - pad_y) / ratio

        x1_unpadded = max(0, min(orig_w, x1_unpadded))
        y1_unpadded = max(0, min(orig_h, y1_unpadded))
        x2_unpadded = max(0, min(orig_w, x2_unpadded))
        y2_unpadded = max(0, min(orig_h, y2_unpadded))

        scaled_boxes.append((x1_unpadded, y1_unpadded, x2_unpadded, y2_unpadded, score, class_id))

    return scaled_boxes


def draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    for x1, y1, x2, y2, score, class_id in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        class_name = CLASS_NAMES.get(class_id, "Неизвестно")
        text = f"{class_name} ({score:.2f})"

        text_bbox = draw.textbbox((x1, y1 - 10), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [x1, y1 - text_height - 5, x1 + text_width + 5, y1],
            fill=color
        )
        draw.text((x1 + 2, y1 - text_height - 3), text, fill=(0, 0, 0))

    return image_copy

app = FastAPI()

model = load_model()

@app.post("/detect")
async def detect_parking(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    boxes = detect_parking_spaces(image, model, conf_threshold=0.3)
    result_image = draw_boxes(image, boxes)

    buffer = io.BytesIO()
    result_image.save(buffer, format='PNG')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)