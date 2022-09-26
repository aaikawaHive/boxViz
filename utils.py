import cv2
import json
import os.path as osp
import random

labels = ["ball", "car", "cup_bottle_can", "golf_club", "hat", "headphones", "helmets", "jersey", "shoe", "watches", "static_sign", "home_plate_sign", "stadium_front", "aerial_stadium", "dugout", "player_fan_tunnel", "media_backdrop", "playing_area", "jumbotron_screen", "fan_area", "lower_level_banner", "bball_stanchion", "on_court_seating", "upper_level_banner", "basketball_pole_pad", "field_goal_post", "football_team_bench"]

def format_from_txt(filepath):
    annotations = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            gt = line.split()
            annotations.append(gt)
            line = f.readline()
    return annotations 

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

def format_from_hsl_swin(filepath):
    response = json.load(open(filepath, 'r', encoding='utf-8'))['response']['output'][0]['rect']
    return_list = []

    for item in response:
        x0 = item['box']['dimensions']['left']
        y0 = item['box']['dimensions']['top']
        x1 = item['box']['dimensions']['right']
        y1 = item['box']['dimensions']['bottom']

        text = item['detection_classes']['list'][0]['class']
        score = item['detection_classes']['list'][0]['score']

        return_list.append([text, score, x0, y0, x1, y1])
    return return_list

def format_from_torch(filepath):
    import torch
    data = torch.load(filepath)
    return_list = []
    boxes = data.pred_boxes.tensor.detach().cpu()
    scores = data.scores.detach().cpu()
    texts = data.pred_classes.detach().cpu()

    for box, score, text in zip(boxes, scores, texts):
        x0, y0, x1, y1 = box
        score = score.item()
        text = text.item()
        text = labels[text]

        return_list.append([text, score, x0, y0, x1, y1])
    return return_list

def get_groundtruths(filepath): # from detectron format
    full_list = json.load(open(filepath, 'r'))
    return {osp.basename(item['file_name']) : item['annotations'] for item in full_list}
