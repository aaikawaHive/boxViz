from flask import Flask
import os
import cv2
from boxViz.data import *
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

################################################################################################################################################

if GROUNDTRUTHS:
    groundtruths = {}
    for gt in GROUNDTRUTHS:
        groundtruths.update(get_groundtruths(gt))
else:
    groundtruths = None

if PREDICTIONS:
    all_preds = {}
    for filepath, formatter in PREDICTIONS.values():
        all_preds.update(get_preds(filepath, formatter))
else:
    all_preds = None

im_list = [im for im in sorted(os.listdir(IMAGES))]
table = [
    {
        'name' : img,
    } 
    for img in im_list]
img2idx = {img : i for i, img in enumerate(im_list)}
idx2img = {i : img for i, img in enumerate(im_list)}
active_filters = ""

pager = Pager(len(table))

app = Flask(__name__, static_folder='static/')
app.config.update(
    APPNAME=APPNAME,
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
