"""
Utilities for formatting and reading prediction/groundtruth files.  return value expects [label, score, x0, y0, x1, y1]
"""
import json
import torch
import os.path as osp
import os
from functools import cache
from tqdm import tqdm

@cache
def load_cached(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_from_txt(filepath, labels=None):
    filepath += '.txt'
    annotations = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            gt = line.split()
            annotations.append(gt)
            line = f.readline()
    return annotations 

def format_from_hsl_swin(filepath, labels=None):
    filepath += '.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        response = json.load(f)['response']['output'][0]['rect']
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

def format_from_hsl_triton(filepath, labels=None):
    filepath += '.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        response = json.load(f)['response']['output'][0]
    return_list = []

    for item in response:
        x0 = item['bbox']['left']
        y0 = item['bbox']['top']
        x1 = item['bbox']['right']
        y1 = item['bbox']['bottom']

        text = item['class']
        score = item['score']

        return_list.append([text, score, x0, y0, x1, y1])
        
    return return_list

def format_from_video_hsl_torch(filepath, labels=None):
    """
    grab frame from video specified in the filepath.
    Convention is that the image files should be named {video_name}_%08d.jpg so you can always grab the frame number
    """
    directory, file_name = osp.split(filepath)
    video_name, frame_no = file_name[:-13], int(file_name[-12:-4])
    file_name = video_name + '.json'
    filepath = directory + '/' + file_name
    response = load_cached(filepath)['response']['output'][frame_no]
    response = response['rect']
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

def format_from_video_hsl_triton(filepath, labels=None):
    """
    grab frame from video specified in the filepath.
    Convention is that the image files should be named {video_name}_%08d.jpg so you can always grab the frame number
    """
    directory, file_name = osp.split(filepath)
    video_name, frame_no = file_name[:-13], int(file_name[-12:-4])
    file_name = video_name + '.json'
    filepath = directory + '/' + file_name
    response = load_cached(filepath)['response']['output'][frame_no]
    return_list = []
    for item in response:
        x0 = item['bbox']['left']
        y0 = item['bbox']['top']
        x1 = item['bbox']['right']
        y1 = item['bbox']['bottom']

        text = item['class']
        score = item['score'][0]

        return_list.append([text, score, x0, y0, x1, y1])
    return return_list

def format_from_torch(filepath, labels=None):
    assert labels is not None
    filepath += '.pt'
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

        return_list.append([text, score, x0.item(), y0.item(), x1.item(), y1.item()])
    return return_list

def get_groundtruths(filepath): # from detectron format
    full_list = json.load(open(filepath, 'r'))
    return {osp.basename(item['file_name']) : item['annotations'] for item in full_list}

def get_preds(detections, formatter):
    ret = {}
    count = 0
    for anno_file in tqdm(os.listdir(detections)):
        img_name = osp.splitext(anno_file)[0]
        path = osp.join(detections, img_name)
        try:
            ret[img_name] = formatter(path)
        except json.decoder.JSONDecodeError:
            print(f'missing prediction for {path}')
            count += 1
            continue
    print(f'total missing = {count}')
    return ret
