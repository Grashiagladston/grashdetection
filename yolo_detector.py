from ultralytics import YOLO
from PIL import Image


# ============================================================
# DATA CLASSES
# ============================================================

class ObjectDetection:

    def __init__(self, box_2d, label, is_target):

        self.box_2d = box_2d
        self.label = label
        self.is_target = is_target


class DetectionResult:

    def __init__(self, objects):

        self.objects = objects


# ============================================================
# YOLO DETECTOR
# ============================================================

class GeminiDetector:
    """
    Drop-in replacement for the original GeminiDetector.
    Uses YOLO internally while keeping the same interface.
    """

    def __init__(self):

        # Small, fast model suitable for Render
        self.model = YOLO("yolo11n.pt")

    def detect(self, image, target_object):

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        results = self.model(image, verbose=False)

        objects = []

        target = target_object.strip().lower()

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                label = self.model.names[cls]

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                obj = ObjectDetection(
                    box_2d=[ymin, xmin, ymax, xmax],
                    label=label,
                    is_target=(label.lower() == target)
                )

                objects.append(obj)

        return DetectionResult(objects)
