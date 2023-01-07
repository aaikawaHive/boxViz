from flask import Flask
import os
from collections import defaultdict
from functools import reduce
from boxViz.data import get_groundtruths, get_preds
from boxViz.pager import Pager
from boxViz import APPNAME, IMAGES, GROUNDTRUTHS, PREDICTIONS, LABELS, label_filters

def init_groundtruth(gts):
    """
    returns a dictionary mapping 'filename' ----> 'annotations'
    """
    groundtruths = {}
    if len(gts) == 0: 
        for img in os.listdir(IMAGES):
            groundtruths[img] = []
        return groundtruths

    for gt in gts:
        groundtruths.update(get_groundtruths(gt))
    return groundtruths

def init_label_set(groundtruths):
    """
    returns dictionary mapping from 'label' to Set[str] of filenames
    which correspond to files containing this label
    """
    label_set = defaultdict(set)
    for filename, annotations in groundtruths.items():
        for anno in annotations:
            label_set[anno['saved_label']].add(filename)
    return label_set

def update(image_names):
    """
    mutates im_list, img2idx, idx2img, and pager to
    use the filtered list in im_list
    """
    global im_list, img2idx, idx2img, pager
    im_list = sorted([im for im in image_names])
    img2idx = {img : i for i, img in enumerate(im_list)}
    idx2img = {i : img for i, img in enumerate(im_list)}
    pager = Pager(len(im_list))

def filter_images(groundtruth):
    """
    filters the im_list, img2idx, and idx2img so that only instances with
    groundtruths labels in `label_filters` are included
    """
    included_images = set(groundtruths.keys())
    if len(label_filters) == 0:
        update(included_images)
        return

    included_images = None
    # filter by label
    for label in label_filters:
        if included_images is None:
            included_images = label_set[label]
        else:
            included_images = included_images.union(label_set[label])
    # filter out images that are not included in groundtruth labels
    included_images = included_images.intersection(set(groundtruths.keys()))
    update(included_images)
    return

groundtruths = init_groundtruth(GROUNDTRUTHS)
label_set = init_label_set(groundtruths)
im_list, img2idx, idx2img, pager = None, None, None, None
filter_images(groundtruths)

app = Flask(__name__, static_folder='static/')
app.config.update(
    APPNAME=APPNAME,
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
