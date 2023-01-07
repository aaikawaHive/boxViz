from boxViz.data import format_from_hsl_triton, format_from_hsl_swin

APPNAME = "BoxViz"
IMAGES = '/persist/aaikawa/logo_objects_sport_location/images/' # list of images
# IMAGES = '/persist/aaikawa/hsl_scripts/train_images/' # list of images
# IMAGES = '/persist/aaikawa/hsl_scripts/INP_shubhang/' # list of images
# IMAGES = '/persist/aaikawa/hsl_scripts/INP_defender/' # list of images
# IMAGES = '/persist/aaikawa/brand_dumps/test_images/' # list of images
# IMAGES = '/persist/aaikawa/datasets/coco/val2017/' # list of images
GROUNDTRUTHS = [
        # '/persist/aaikawa/brand_dumps/main_logo_test_detectron2.json',
#    '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_learfield_test_detectron2.json', # detectron2 annotations format
    '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_sports_location_test_detectron2.json', # detectron2 annotations format
#   '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_logo_objects_test_detectron2.json', # detectron2 annotations format
    # '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_learfield_test_detectron2_fixed.json', # detectron2 annotations format
    # '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_sports_location_test_detectron2.json', # detectron2 annotations format
    # '/persist/aaikawa/logo_objects_sport_location/logo_locations_27_logo_objects_test_detectron2.json', # detectron2 annotations format
] 
PREDICTIONS = { # dictionary should be 'model_name' : Tuple( 'predictions_folder_path', 'prediction_formatter' )
    'dino' : ('/persist/aaikawa/hsl_scripts/TEST_dino_nms50', format_from_hsl_triton),
    'swin' : ('/persist/aaikawa/hsl_scripts/TEST_swin', format_from_hsl_triton),
    # 'swin' : ('/persist/aaikawa/hsl_scripts/shubhang', format_from_hsl_swin),
    # 'swin' : ('/persist/aaikawa/hsl_scripts/TEST_train_images', format_from_hsl_swin),
    # 'swin' : ('/persist/aaikawa/hsl_scripts/defender', format_from_hsl_swin),
}
# LABELS = '/persist/aaikawa/main_logo/classes.labels'
LABELS = [] #open(LABELS, 'r').read().split()
label_filters = []
