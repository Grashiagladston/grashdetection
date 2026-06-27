from ultralytics import YOLO
from PIL import Image


class GeminiDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, image, target_object):

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        results = self.model(image)

        detections = []

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])

                label = self.model.names[cls]

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                detections.append({
                    "label": label,
                    "is_target": label.lower() == target_object.lower(),
                    "box_2d": [ymin, xmin, ymax, xmax]
                })

        class Result:
            def __init__(self, objects):
                self.objects = objects

        return Result(detections)
