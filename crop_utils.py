from PIL import Image


def crop_detected_objects(image, detections, target_object):

    cropped_images_data = []

    target_object_lower = target_object.strip().lower()

    for obj in detections.objects:

        is_match = (
            obj.is_target
            or target_object_lower in obj.label.lower()
            or obj.label.lower() in target_object_lower
        )

        if not is_match:
            continue

        ymin, xmin, ymax, xmax = obj.box_2d

        crop = image.crop(
            (
                xmin,
                ymin,
                xmax,
                ymax
            )
        )

        cropped_images_data.append(
            {
                "image": crop,
                "label": obj.label,
                "box_2d": obj.box_2d
            }
        )

    return cropped_images_data
