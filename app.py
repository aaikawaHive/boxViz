import base64
import csv
import cv2
from flask import Flask, render_template, request, redirect, url_for, make_response
import os
import requests
import tempfile
from pager import Pager
from utils import get_groundtruths, plot_one_box, format_from_hsl_swin, format_from_torch

APPNAME = "BoxViz"
IMAGES = '/persist/aaikawa/data/data/'
GROUNDTRUTHS = [
    '/persist/aaikawa/data/split_test_0_detectron2.json', # should be detectron2 format
    '/persist/aaikawa/data/split_test_1_detectron2.json',
    '/persist/aaikawa/data/split_test_2_detectron2.json',
    '/persist/aaikawa/data/split_test_3_detectron2.json',
]
# PREDICTIONS = '/persist/aaikawa/hsl_scripts/TEST_dino/' # should be hsl format
PREDICTIONS = '/persist/aaikawa/preds/'

def plot_boxes(filepath, showGT=True, showPred=True, thresh=0.1):
    top, bottom, left, right = 30, 0, 30, 0
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None: return None
    basename = os.path.basename(filepath)
    
    if showGT:
        if not basename in groundtruths:
            print(f'{basename} does not have groundtruth')
        else:
            annotations = groundtruths[basename]
            for gt in annotations:
                print(gt)
                label = gt['saved_label']
                x = gt['bbox']
                img = plot_one_box(x, img, color=(0, 0, 0), label=f'GT : {label}', line_thickness=2)
    if showPred and PREDICTIONS is not None:
        # annotations = format_from_hsl_swin(PREDICTIONS + basename + '.json')
        annotations = format_from_torch(PREDICTIONS + basename + '.pt')
        print("!" * 10 ** 2)
        for label, score, *x in annotations:
            if score > int(thresh) / 100.0:                
                print(label, score, x)
                print()
                label = f'{label} : {score:.02f}'
                img = plot_one_box(x, img, label=label, line_thickness=1)
        print("!" * 10 ** 2)
    return img

groundtruths = {}
for gt in GROUNDTRUTHS:
    groundtruths.update(get_groundtruths(gt))

# TODO : handle filenames with special chars (i.e. %)
im_list = [im for im in os.listdir(IMAGES) if im in groundtruths and not '%' in im]
table = [{'name' : img} for img in im_list]
img2idx = {img : i for i, img in enumerate(im_list)}
idx2img = {i : img for i, img in enumerate(im_list)}


pager = Pager(len(table))

app = Flask(__name__, static_folder='static/')
app.config.update(
    APPNAME=APPNAME,
    )

@app.route('/')
def index():
    return redirect(f'/{im_list[0]}')

@app.route('/<filename>/', methods=['POST', 'GET'])
def image_view(filename):
    
    if request.method == 'POST':
        print(request.form)
        showGT, showPred = 'showGT' in request.form, 'showPred' in request.form
        thresh = int(request.form['myRange'])
    else:
        showGT, showPred, thresh = True, True, 10
    image = plot_boxes(IMAGES + filename, showGT=showGT, showPred=showPred, thresh=thresh)
    if image is None:
        return render_template("404.html", file_path=IMAGES + filename, pager=pager), 404
    print(filename)
    ind = img2idx[filename]
    pager.current = ind
    cv2.imwrite('static/temp.jpg', image)

    response = render_template(
        'imageview.html',
        index=ind,
        pager=pager,
        data=table[ind],
        request=request,
        )
    return response

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
