from ultralytics import YOLO
from PIL import Image


class ObjectDetection:
    def __init__(self, box_2d, label, is_target):
        self.box_2d = box_2d
        self.label = label
        self.is_target = is_target


class DetectionResult:
    def __init__(self, objects):
        self.objects = objects


class GeminiDetector:
    """
    Drop-in replacement for the old GeminiDetector.
    Uses YOLOv8 internally but returns the same object structure.
    """

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, image, target_object):

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        results = self.model(image)

        objects = []

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                label = self.model.names[cls]

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                obj = ObjectDetection(
                    box_2d=[ymin, xmin, ymax, xmax],
                    label=label,
                    is_target=(label.lower() == target_object.lower())
                )

                objects.append(obj)

        return DetectionResult(objects)
