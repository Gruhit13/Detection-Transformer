import os
import numpy as np
import pandas as pd

from config import CLASSES_TO_LABEL
from DETRDataset import DETRDataset

from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

def form_data(path):
	files = os.listdir(path)
	data_file = None

	# Extract the '.csv' file that contains bounding box info and class labels
	for fname in files:
	  ext = fname.split('.')[-1]

	  if ext == 'csv':
		data_file = os.path.join(path, fname)
		break

	image_data = pd.read_csv(data_file)

	image_data['class'] = image_data['class'].apply(lambda x: CLASSES_TO_LABEL[x])
	image_data['filename'] = image_data['filename'].apply(lambda x: os.path.join(path, x))
	
	return image_data

def groupClassBbox(image_data):
	data = image_data.copy()

	file_class_bbox = {}

	for ind, item in data.iterrows():
		fname = item['filename']
		if fname not in file_class_bbox:
		file_class_bbox[fname] = {
			  'class' : [item['class']],
			'bbox' : [[item['xmin'], item['ymin'], item['xmax'], item['ymax']]]
		}
		else:
		file_class_bbox[fname]['class'].append(item['class'])
		file_class_bbox[fname]['bbox'].append([item['xmin'], item['ymin'], item['xmax'], item['ymax']])

	new_data = pd.DataFrame.from_dict(file_class_bbox, orient='index')
	new_data = new_data.rename_axis('filename').reset_index()
	return new_data

def collate_fn(batch):
	"""This batch collate function to stack batches from customer torch dataset class"""
	images = []
	masks = []

	targets = []
	for input, target in batch:

	images.append(input['img'])
	masks.append(input['mask'])
	targets.append(target)

	return torch.stack(images), torch.stack(masks), targets

def getDataset(dataset_path, num_queries, batch_size, image_dim=512):
	image_data = form_data(dataset_path)
	dataset = groupClassBbox(image_data)

	detr_dataset = DETRDataset(
						dataset, 
						CLASSES_TO_LABEL, 
						num_queries, 
						T.Compose([T.ToTensor(), T.Resize([image_dim, image_dim], antialias=False)])
					)

	validation_size = int(0.1 * detr_dataset.__len__())

	generator = torch.Generator().manual_seed(42)
	train_data, val_data = random_split(
						detr_dataset, 
						[detr_dataset.__len__() - validation_size, validation_size], 
						generator=generator
					)

	train_loader = DataLoader(
						train_data, 
						batch_size=batch_size, 
						drop_last=True, 
						collate_fn=collate_fn, 
						shuffle=True
					)
	
	val_loader = DataLoader(
						val_data, 
						batch_size=batch_size, 
						drop_last=True, 
						collate_fn=collate_fn
					)

	return train_loader, val_loader