from boxViz.data import format_from_hsl_triton, format_from_hsl_swin

APPNAME = "BoxViz"
IMAGES = '/persist/aaikawa/main_logo/main_logo_images/' # list of images
GROUNDTRUTHS = [
    '/persist/aaikawa/main_logo/main_logo_test_detectron2.json', # detectron2 annotations format
] 
PREDICTIONS = { # dictionary should be 'model_name' : Tuple( 'predictions_folder_path', 'prediction_formatter' )
    'triton' : ('/persist/aaikawa/hsl_scripts/TEST_triton', format_from_hsl_triton),
    'torch' : ('/persist/aaikawa/hsl_scripts/TEST_torch', format_from_hsl_swin)
}
LABELS = '/persist/aaikawa/main_logo/classes.labels'
LABELS = open(LABELS, 'r').read().split()
label_filters = ['aral']