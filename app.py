import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, request, send_file
import torch
from PIL import Image, ImageDraw
import io
import numpy as np

app = Flask(__name__)

chest_model = torch.hub.load('yolov5', 'custom', path='chest_detector.pt', source='local')
wound_model = torch.hub.load('yolov5', 'custom', path='wound_detector.pt', source='local')
burn_model  = torch.hub.load('yolov5', 'custom', path='burn_classifier.pt', source='local')

chest_model.conf = 0.25
wound_model.conf = 0.10
burn_model.conf  = 0.50

chest_model.iou = 0.40
wound_model.iou = 0.40
burn_model.iou  = 0.40

def get_wound_info(name, conf):
    types = {
        'wound': 'Laceration', 'cut': 'Cut', 'bruise': 'Bruise',
        'abrasion': 'Abrasion', 'puncture': 'Puncture', 'bleeding': 'Open Bleeding',
    }
    wtype    = types.get(name.lower(), 'Open Wound')
    severity = 'SEVERE' if conf >= 0.65 else 'MODERATE' if conf >= 0.35 else 'MINOR'
    return wtype, severity

def get_burn_info(name, conf):
    types = {'0': '1st Degree', '1': '2nd Degree', '2': '3rd Degree'}
    btype    = types.get(str(name), 'Burn Injury')
    severity = 'SEVERE' if conf >= 0.65 else 'MODERATE' if conf >= 0.35 else 'MINOR'
    return btype, severity

def iou_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6)

def deduplicate(boxes, thresh=0.45):
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    kept  = []
    for box in boxes:
        if not any(iou_overlap(
            (box['x1'], box['y1'], box['x2'], box['y2']),
            (k['x1'],   k['y1'],  k['x2'],  k['y2'])) > thresh for k in kept):
            kept.append(box)
    return kept

def is_fullimage_box(x1, y1, x2, y2, img_w, img_h, thresh=0.60):
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    return (box_area / img_area) > thresh

def run_multiscale(model, img, scales=[416, 640]):
    orig_w, orig_h = img.size
    all_boxes = []
    for size in scales:
        resized = img.resize((size, size))
        results = model(resized).pandas().xyxy[0]
        sx, sy  = orig_w / size, orig_h / size
        for _, row in results.iterrows():
            x1 = int(row['xmin'] * sx)
            y1 = int(row['ymin'] * sy)
            x2 = int(row['xmax'] * sx)
            y2 = int(row['ymax'] * sy)
            if is_fullimage_box(x1, y1, x2, y2, orig_w, orig_h):
                continue
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            all_boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': float(row['confidence']),
                'name': str(row['name']),
            })
    return deduplicate(all_boxes)

def detect_red_regions(img):
    arr = np.array(img)
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    blood_mask = ((r - g > 50) & (r - b > 50) & (r > 120)).astype(np.uint8)
    h, w  = blood_mask.shape
    cell  = 80
    boxes = []
    for gy in range(0, h - cell, cell // 2):
        for gx in range(0, w - cell, cell // 2):
            patch   = blood_mask[gy:gy+cell, gx:gx+cell]
            density = patch.sum() / (cell * cell)
            if density > 0.30:
                x1 = max(0, gx - 10)
                y1 = max(0, gy - 10)
                x2 = min(w, gx + cell + 10)
                y2 = min(h, gy + cell + 10)
                if is_fullimage_box(x1, y1, x2, y2, w, h):
                    continue
                boxes.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': float(density),
                    'name': 'bleeding',
                })
    return deduplicate(boxes, thresh=0.30)

def draw_label(draw, x1, y1, x2, y2, lines, color, img_w, img_h):
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    line_h = 22
    pad    = 6
    box_h  = line_h * len(lines) + pad * 2
    box_w  = max(len(l) for l in lines) * 9 + pad * 2
    lx = max(0, min(x1, img_w - box_w))
    ly = y1 - box_h if y1 - box_h >= 0 else y2
    draw.rectangle([lx+2, ly+2, lx+box_w+2, ly+box_h+2], fill='black')
    draw.rectangle([lx,   ly,   lx+box_w,   ly+box_h  ], fill=color)
    for i, line in enumerate(lines):
        draw.text((lx + pad, ly + pad + i * line_h),
                  line, fill='yellow' if i == 0 else 'white')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    img  = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    detected_any = False

    # CHEST
    for box in run_multiscale(chest_model, img):
        detected_any = True
        lines = [
            "BODY PART",
            "Part: " + box['name'],
            "Conf: " + str(int(box['conf'] * 100)) + "%",
        ]
        draw_label(draw, box['x1'], box['y1'], box['x2'], box['y2'],
                   lines, '#1565C0', img_w, img_h)

    # WOUND
    wound_boxes = run_multiscale(wound_model, img)
    if not wound_boxes:
        wound_boxes = detect_red_regions(img)

    for box in wound_boxes:
        detected_any = True
        wtype, severity = get_wound_info(box['name'], box['conf'])
        color = '#B71C1C' if severity == 'SEVERE' else '#E53935' if severity == 'MODERATE' else '#C62828'
        lines = [
            "WOUND DETECTED",
            "Type: " + wtype,
            "Severity: " + severity,
            "Conf: " + str(int(box['conf'] * 100)) + "%",
        ]
        draw_label(draw, box['x1'], box['y1'], box['x2'], box['y2'],
                   lines, color, img_w, img_h)

    # BURN
    for box in run_multiscale(burn_model, img):
        detected_any = True
        btype, severity = get_burn_info(box['name'], box['conf'])
        color = '#E65100' if severity == 'SEVERE' else '#FB8C00' if severity == 'MODERATE' else '#FF8F00'
        lines = [
            "BURN DETECTED",
            "Type: " + btype,
            "Severity: " + severity,
            "Conf: " + str(int(box['conf'] * 100)) + "%",
        ]
        draw_label(draw, box['x1'], box['y1'], box['x2'], box['y2'],
                   lines, color, img_w, img_h)

    if not detected_any:
        msg = "No injuries detected"
        draw.rectangle([8, 8, len(msg) * 10 + 16, 36], fill='#424242')
        draw.text((12, 12), msg, fill='white')

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/')
def home():
    return "Medical AI App is running!"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)