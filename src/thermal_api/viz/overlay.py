
import cv2 as cv

COLOR_BODY = (0, 255, 0)
COLOR_HEAD = (0, 255, 255)
COLOR_FOOT = (255, 255, 0)
COLOR_LOW  = (255, 0, 0)
COLOR_HIGH = (0, 165, 255)
COLOR_FIRE = (0, 0, 255)

def draw_boxes(img, boxes, color, thickness=1):
    for (x1, y1, x2, y2, *_) in boxes:
        cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_human(img, result):
    bodies = [(b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3], b["area"]) for b in result["bodies"]]
    draw_boxes(img, bodies, COLOR_BODY, 1)
    for h in result["heads"]:
        x, y, r = h.get("circle", (None, None, None))
        if x is not None:
            cv.circle(img, (x, y), r, COLOR_HEAD, 1)

def draw_footprints(img, result):
    foot = [(f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3], f["area"]) for f in result["footprints"]]
    draw_boxes(img, foot, COLOR_FOOT, 1)

def draw_abnormal(img, result):
    for name, color in ("low", COLOR_LOW), ("high", COLOR_HIGH), ("fire", COLOR_FIRE):
        boxes = [(b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3], b["area"]) for b in result.get(name, [])]
        draw_boxes(img, boxes, color, 1)
