DATA_FOLDER = './data'
TEST_FOLDER = DATA_FOLDER + '/' + 'test'   	# folder with testing images
TRAIN_FOLDER = DATA_FOLDER + '/' + 'train' 	# folder with training images

MODEL_FOLDER = './model'

RESULT_FOLDER = './result'

IMAGE_HEIGHT = 64  # height of the image
IMAGE_WIDTH = 64   # width of the image
NUM_CHANNELS = 1   # number of channels of the image

INPUT_SIZE = 64

SAVE_SIZE = 5

LEARNING_RATES = [1e-2, 5.5e-3, 1e-3]
LEARNING_RATE_DECAY = 0.95

NUM_EPOCHS = 50

HALF_BATCH_SIZE = 32