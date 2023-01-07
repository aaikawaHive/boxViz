import cv2
from flask import render_template, request, redirect
import os
import requests
from boxViz.pager import Pager
from boxViz.plot import plot_one_box, plot_boxes
from boxViz.app import app, groundtruths, im_list, img2idx, idx2img, pager
from boxViz import PREDICTIONS, LABELS, IMAGES
from boxViz.data import get_preds

directory = os.path.dirname(__file__)
labelFilter = ''
thresh = 0.01

def imview(filename):
    """
    Renders template for imageviewer. Called by `image_percent`, `image_view`, `image_view_idx`
    which handle url routing.
    """
    # handle user input
    global labelFilter, thresh
    if request.method == 'POST':
        try:
            thresh = float(request.form['confidenceFilter'])
        except ValueError:
            print('{} is not a float. Setting to default 0.01'.format(request.form['confidenceFilter']))
            thresh = 0.01
        labelFilter = request.form['labelFilter']
    
    gts = groundtruths[filename]
    preds = PREDICTIONS 

    kwargs = {
        'thresh' : thresh,
        'labels' : LABELS,
        'filters' : labelFilter
    }

    # plot preds and gts together
    filepath = os.path.join(IMAGES, filename)
    image, annotations, original_dims = plot_boxes(filepath, groundtruths=gts, predictions=preds, plot_gt=True, **kwargs)
    if image is None:
        return render_template("404.html", file_path=filepath, pager=pager), 404
    
    ind = img2idx[filename]
    pager.current = ind
    assert cv2.imwrite(os.path.join(directory, 'static/temp.jpg'), image)

    # make seperate panels for preds and gts
    all_boxes = {}
    if groundtruths:
        filepath = os.path.join(IMAGES, filename)
        image, annotations, original_dims = plot_boxes(filepath, groundtruths=gts, **kwargs, plot_gt=True) # just gts
        assert cv2.imwrite(os.path.join(directory, 'static/temp_gt.jpg'), image)
        all_boxes['gt'] = annotations

    # one image per model
    for k, v in PREDICTIONS.items():
        filepath = os.path.join(IMAGES, filename)
        image, annotations, original_dims = plot_boxes(filepath, groundtruths=gts, predictions={k : v}, **kwargs, plot_gt=False) 
        all_boxes[f'{k}'] = annotations
        assert cv2.imwrite(os.path.join(directory, f'static/temp_{k}.jpg'), image)

    table = {
        'name' : filename,
        'dims' : original_dims,
    }
    response = render_template(
        'imageview.html',
        pager=pager,
        data=table,
        request=request,
        all_boxes=all_boxes,
        labelFilter=labelFilter,
        confidenceFilter=thresh,
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
    """
    used by Pager
    """
    # filename = idx2img[int(request.form['index'])]
    # return redirect('/' + filename)
    return redirect('/' + request.form['index'])
