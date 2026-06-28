from PIL import Image, ImageDraw, ImageFont


def draw_annotations(image, detections, target_object):

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    target_count = 0
    boxes_list = []

    target_object_lower = target_object.strip().lower()

    box_color = "#00F0FF"

    for obj in detections.objects:

        is_match = (
            obj.is_target
            or target_object_lower in obj.label.lower()
            or obj.label.lower() in target_object_lower
        )

        if not is_match:
            continue

        target_count += 1

        ymin, xmin, ymax, xmax = obj.box_2d

        boxes_list.append({
            "index": target_count,
            "label": obj.label,
            "box_2d": obj.box_2d
        })

        left = xmin
        top = ymin
        right = xmax
        bottom = ymax

        draw.rectangle(
            [left, top, right, bottom],
            outline=box_color,
            width=5
        )

        label_text = f"{obj.label.upper()} #{target_count}"

        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w = len(label_text) * 8
            text_h = 18

        label_top = max(0, top - text_h - 8)

        draw.rectangle(
            [left, label_top, left + text_w + 12, label_top + text_h + 8],
            fill=box_color
        )

        draw.text(
            (left + 6, label_top + 4),
            label_text,
            fill="black",
            font=font
        )

    return annotated_image, target_count, boxes_list
