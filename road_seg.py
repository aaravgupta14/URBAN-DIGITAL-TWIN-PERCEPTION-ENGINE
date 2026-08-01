import cv2
import torch
import numpy as np
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
).to(device)

model.eval()

ROAD_CLASS = 0

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)


def segment_road(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(
        images=rgb,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False
    )

    pred = logits.argmax(dim=1)[0].cpu().numpy()

    road_mask = np.uint8(pred == ROAD_CLASS) * 255

    road_mask[:frame.shape[0] // 10, :] = 0

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        road_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(road_mask)

    hull = None
    largest = None

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        hull = cv2.convexHull(largest)

        cv2.drawContours(
            clean_mask,
            [hull],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return clean_mask, hull, largest
