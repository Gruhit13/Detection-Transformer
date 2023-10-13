NUM_QUERIES = 5
BATCH_SIZE = 8
IMG_CHANNELS = 3
LAYERS = [2, 2, 2, 2]
NUM_CLASSES = 11 # Using Udacity Cars dataset that has 11 labels
CHANNELS = [64, 128, 256, 256]
IMG_DIM = 512 # This is the image dimension we will train our model for

D_MODEL = 256
NUM_HEADS = 8
FFN_DIMS = 2048
N_LAYERS = 2
PRECISION = 16

NUM_POS_FT = D_MODEL // 2

WEIGHT_KEYS = {
    'label_loss': 1,
    'l1_loss': 5,
    'giou_loss': 2
}

EPOCHS = 300

# This is the path to the dataset
DATASET_PATH = './udacity-self-driving-car-dataset/data/export'
LOG_DIR = './logs'

# Create a label for each unique class. We are going to use udacity self driving car dataset
# hence this labels are suitable for it. If you use any other dataset then infer accordingly
CLASSES_TO_LABEL = {
     'trafficLight-RedLeft': 0,
     'trafficLight': 1,
     'trafficLight-GreenLeft': 2,
     'pedestrian': 3,
     'trafficLight-YellowLeft': 4,
     'truck': 5,
     'car': 6,
     'biker': 7,
     'trafficLight-Yellow': 8,
     'trafficLight-Green': 9,
     'trafficLight-Red': 10
}

LABELS_TO_CLASSES = {v:k for k, v in CLASSES_TO_LABEL.items()}