from easydict import EasyDict as edict
cfg = edict()
cfg.TRAIN_FILE = './test_set.txt'
cfg.BATCH_SIZE = 32
cfg.NUM_CHANNELS = 3
cfg.LEARNING_RATE = 0.001
cfg.IMAGE_WIDTH = 227 
cfg.IMAGE_HEIGHT = 227
