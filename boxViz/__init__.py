from boxViz.data import format_from_hsl_triton, format_from_hsl_swin

APPNAME = "BoxViz"
IMAGES = '/path/to/images_folder'
GROUNDTRUTHS = [
   '/path/to/annotations.json', # detectron2 annotations format
] 
PREDICTIONS = { # dictionary should be 'model_name' : Tuple( 'predictions_folder_path', 'prediction_formatter' )
    'my_model' : ('/path/to/predictions/', format_from_hsl_triton),
}

LABELS = [] #open(LABELS, 'r').read().split()
label_filters = []
