import cv2
import os.path as osp
import random

def shift_bbox(bbox, delta):
    """
    Shifts a bbox [x0, y0, x1, y1] by delta [∆x, ∆y]
    """
    bbox = bbox[:]
    bbox[0] += delta[0]
    bbox[2] += delta[0]
    bbox[1] += delta[1]
    bbox[3] += delta[1]
    return bbox

def plot_boxes(filepath, groundtruths=None, predictions=None, thresh=0.1, labels=None, filters=""):
    # pad the top so you can read labels that take the whole frame
    label_set = set(filters.split())
    top, bottom, left, right = 10, 0, 0, 0
    delta = [left, top]
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if img is None: return None
    basename = osp.basename(filepath)
    ret = []
    if groundtruths:
        annotations = groundtruths[basename]
        for i, gt in enumerate(annotations):
            label = gt['saved_label']
            if len(label_set) == 0 or label in label_set:
                x = shift_bbox(gt['bbox'], delta)
                img = plot_one_box(x, img, color=(0, 0, 0), label=f'GT : {label}', line_thickness=2)
                ret.append([label] + [f'{c:.01f}' for c in x])
    if predictions:
        for model_name, (pred_folder, formatter) in predictions.items():
            annotations = formatter(osp.join(pred_folder, basename), labels=labels)
            for label, score, *x in annotations:
                if score > int(thresh) / 100.0 and (len(label_set) == 0 or label in label_set):                
                    text = f'{label[:8]} : {score:.01f}'
                    x = shift_bbox(x, delta)
                    img = plot_one_box(x, img, label=text, line_thickness=1)
                    ret.append([label, f'{score:.01f}'] + [f'{c:.01f}' for c in x])
    ret = sorted(ret, key=lambda x: (x[0], -float(x[1])))
    return img, ret

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        img = cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img = cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img
