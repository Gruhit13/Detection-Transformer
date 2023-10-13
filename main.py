from dataset import getDataset
import config

from backbone import ResNet
from criterion import HungarianMatcher, Criterion
from detr import DETR
from positionalEmbedding import PositionalEmbeddingSine
from transformer import Transformer

import lightning.pytorch as pl
from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

def create_detr_model():
	backbone = ResNet(
	    img_channels = config.IMG_CHANNELS,
	    channles= config.CHANNELS,
	    layers = config.LAYERS,
	    expansion = 1,
	    num_classes = config.NUM_CLASSES
	)

	transformer = Transformer(
	    d_model = config.D_MODEL,
	    num_heads = config.NUM_HEADS,
	    ff_dim = config.FFN_DIMS,
	    n_layers = config.N_LAYERS
	)

	positionalEmbedding = PositionalEmbeddingSine(
	    num_pos_ft = config.NUM_POS_FT, 
	    normalize = True
	)

	matcher = HungarianMatcher(
	    cost_class = config.WEIGHT_KEYS['label_loss'],
	    cost_bbox = config.WEIGHT_KEYS['l1_loss'],
	    cost_giou = config.WEIGHT_KEYS['giou_loss']
	)

	criterion = Criterion(
	    matcher = matcher,
	    num_classes = config.NUM_CLASSES,
	    eos_coef = 0.1
	)


	detr = DETR(
	    backbone = backbone,
	    positional_embedding = positionalEmbedding,
	    transformer = transformer,
	    num_classes = config.NUM_CLASSES,
	    num_queries = config.NUM_QUERIES,
	    criterion = criterion,
	    weight_dict = config.WEIGHT_KEYS,
	)

	return detr

def main():

	# Get training and validation dataset
	train_loder, val_loader = getDataset(
			config.DATASET_PATH,
			config.NUM_QUERIES,
			config.BATCH_SIZE
		)

	# Get model
	detr = create_detr_model()


	# Create checkpoints to store best models
	checkpoints = ModelCheckpoint(
			dirpath='./checkpoint',
			monitor='val_total_loss'
		)

	
	# Tensorboard logger to view logs
	loggers = pl_loggers.TensorBoardLogger(config.LOG_DIR)

	# Creating pytorch trainer
	trainer = pl.Trainer(
	    max_epochs = config.EPOCHS,
	    precision = config.PRECISION,
	    logger = loggers,
	    accelerator = 'auto',
	    devices = 'auto',
	    strategy = 'auto',
	    callbacks = [checkpoint_callback]
	)


	# Fitting the model
	trainer.fit(detr, train_loader, val_dataloaders=val_loader)