from flask import Flask
import os
from collections import defaultdict
from functools import reduce
from boxViz.data import get_groundtruths, get_preds, format_from_hsl_triton, format_from_hsl_swin
from boxViz.pager import Pager
################################################################################################################################################
# PROJECT DEFS #################################################################################################################################
################################################################################################################################################

APPNAME = "BoxViz"
IMAGES = '/persist/aaikawa/main_logo/main_logo_images/'
GROUNDTRUTHS = [
    '/persist/aaikawa/main_logo/main_logo_test_detectron2.json', # should be detectron2 format
] # list set of images you want to see
# PREDICTIONS = {
#     'swin' : ('/persist/aaikawa/preds/', format_from_torch)
# } # dict from model name (str) to a set of predictions, where there is one prediction per 
PREDICTIONS = {
    'triton' : ('/persist/aaikawa/hsl_scripts/TEST_triton', format_from_hsl_triton),
    'torch' : ('/persist/aaikawa/hsl_scripts/TEST_torch', format_from_hsl_swin)
}
LABELS = '/persist/aaikawa/main_logo/classes.labels'
LABELS = open(LABELS, 'r').read().split()
label_filters = ['aral']
################################################################################################################################################

def init_groundtruth(gts):
    """
    returns a dictionary mapping 'filename' ----> 'annotations'
    """
    if gts is None: return None
    groundtruths = {}
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

def filter_images():
    """
    filters the im_list, img2idx, and idx2img so that only instances with
    groundtruths labels in `label_filters` are included
    """
    if len(label_filters) == 0:
        images = os.listdir(IMAGES)
        update(images)
        return

    included_images = None
    for label in label_filters:
        if included_images is None:
            included_images = label_set[label]
        else:
            included_images = included_images.union(label_set[label])
    update(included_images)
    return

groundtruths = init_groundtruth(GROUNDTRUTHS)
label_set = init_label_set(groundtruths)
im_list, img2idx, idx2img, pager = None, None, None, None
filter_images()

app = Flask(__name__, static_folder='static/')
app.config.update(
    APPNAME=APPNAME,
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
