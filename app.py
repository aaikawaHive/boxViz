import base64
import csv
import cv2
from flask import Flask, render_template, request, redirect, url_for, make_response
import os
import requests
import tempfile
from pager import Pager
from plot import plot_one_box, plot_boxes
from data import format_from_torch, get_groundtruths, format_from_hsl_triton, get_preds
from box_utils import missing

################################################################################################################################################
# PROJECT DEFS #################################################################################################################################
################################################################################################################################################

APPNAME = "BoxViz"
IMAGES = '/persist/aaikawa/face/test_images/'
GROUNDTRUTHS = None
GROUNDTRUTHS = [
    '/persist/aaikawa/face/face_test_detectron2.json', # should be detectron2 format
] # list set of images you want to see
# PREDICTIONS = {
#     'swin' : ('/persist/aaikawa/preds/', format_from_torch)
# } # dict from model name (str) to a set of predictions, where there is one prediction per 
PREDICTIONS = {
    'triton' : ('/persist/aaikawa/hsl_scripts/TEST_face', format_from_hsl_triton)
}
LABELS = ['face']

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

# TODO : handle filenames with special chars (i.e. %)
# im_list = [im for im in os.listdir(IMAGES) if im in groundtruths and not '%' in im] # ignores files with '%' char
im_list = [im for im in sorted(os.listdir(IMAGES)) if missing(groundtruths, all_preds, im)]
table = [{'name' : img} for img in im_list]
img2idx = {img : i for i, img in enumerate(im_list)}
idx2img = {i : img for i, img in enumerate(im_list)}
active_filters = ""

pager = Pager(len(table))

app = Flask(__name__, static_folder='static/')
app.config.update(
    APPNAME=APPNAME,
    )

def imview(filename):
    if request.method == 'POST':
        showGT, showPred = 'showGT' in request.form, 'showPred' in request.form
        thresh = int(request.form['myRange'])
        active_filters = request.form['labelFilter']
    else:
        showGT, showPred, thresh, active_filters = True, True, 0.1, ''
    gts = groundtruths if showGT else None
    preds = PREDICTIONS if showPred else None

    kwargs = {
        'thresh' : thresh,
        'labels' : LABELS,
        'filters' : active_filters
    }

    # plot preds and gts together
    filepath = os.path.join(IMAGES, filename)
    image, _ = plot_boxes(filepath, groundtruths=gts, predictions=preds, **kwargs)
    if image is None:
        return render_template("404.html", file_path=filepath, pager=pager), 404
    ind = img2idx[filename]
    pager.current = ind
    cv2.imwrite('static/temp.jpg', image)

    # make seperate panels for preds and gts
    all_boxes = {}
    if groundtruths:
        filepath = os.path.join(IMAGES, filename)
        image, annotations = plot_boxes(filepath, groundtruths=gts, **kwargs) # just gts
        cv2.imwrite('static/temp_gt.jpg', image)
        all_boxes['gt'] = annotations

    # one image per model
    for k, v in PREDICTIONS.items():
        filepath = os.path.join(IMAGES, filename)
        image, annotations = plot_boxes(filepath, groundtruths=None, predictions={k : v}, **kwargs) 
        all_boxes[f'{k}'] = annotations
        cv2.imwrite(f'static/temp_{k}.jpg', image)

    response = render_template(
        'imageview.html',
        index=ind,
        pager=pager,
        data=table[ind],
        request=request,
        all_boxes=all_boxes,
        active_filters=active_filters,
        )
    return response

@app.route('/')
def index():
    return redirect(f'/{im_list[0]}')

@app.route('/<project>/<a>/<b>/<c>/<d>/', methods=['POST', 'GET'])
def image_percent(project='', a='', b='', c='', d=''):
    """
    URL handler for files containing %2F
    """
    filename = r'%2F'.join([project, a, b, c, d]) 
    return imview(filename)

@app.route('/<filename>/', methods=['POST', 'GET'])
def image_view(filename=''):
    return imview(filename)

@app.route('/<int:ind>/')
def image_view_idx(ind=None):
    filename = idx2img[ind]
    return redirect('/' + filename)

@app.route('/goto', methods=['POST', 'GET'])    
def goto():
    # filename = idx2img[int(request.form['index'])]
    # return redirect('/' + filename)
    return redirect('/' + request.form['index'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
