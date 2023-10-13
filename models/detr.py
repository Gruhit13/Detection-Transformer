import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Feed Forward Network for bounding box prediction
class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super(MLP, self).__init__()
		self.num_layers = n_layers
		h = [hidden_dim] * (self.num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

		return x

class DETR(pl.LightningModule):
	def __init__(self,
							 backbone,
							 positional_embedding,
							 transformer,
							 num_classes,
							 num_queries,
							 criterion,
							 weight_dict,
							 step_size=200,
							):
		super().__init__()
		self.save_hyperparameters()
		self.backbone = backbone
		self.positional_embedding = positional_embedding
		self.transformer = transformer

		self.criterion = criterion

		# losses weight dictonary
		self.weight_dict = weight_dict

		# LR Scheduler step size
		self.step_size = step_size

		hidden_dim = self.transformer.d_model

		# +1 class if for no-object
		self.class_embed = nn.Linear(hidden_dim, num_classes+1)

		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
		self.query_embed = nn.Embedding(num_queries, hidden_dim)
		self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.step_size)

		return {
				"optimizer": optimizer,
				"lr_scheduler": {
						"scheduler": scheduler,
						"interval": "epoch"
				}
		}

	def __common_step(self, batch, batch_idx):
		images, mask, targets = batch

		src, mask = self.backbone(images, mask)

		pos = self.positional_embedding(src, mask)
		hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

		output_class = self.class_embed(hs)
		output_coord = self.bbox_embed(hs).sigmoid()

		output = {
				"pred_class" : output_class[-1],
				"pred_bbox" : output_coord[-1]
		}

		losses = self.criterion(output, targets)
		loss = self.get_loss(losses)
		return output, losses, loss

	def forward(self, batch):
		images, masks, _ = batch

		src, mask = self.backbone(images, masks)
		pos = self.positional_embedding(src, mask)

		hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

		output_class = self.class_embed(hs)
		output_coord = self.bbox_embed(hs).sigmoid()

		output = {
				"pred_class": output_class[-1],
				"pred_bbox": output_coord[-1]
		}

		return output


	def get_loss(self, losses):
		return sum(self.weight_dict[k] * losses[k] for k in losses.keys())

	def training_step(self, batch, batch_idx):
		output, losses, loss = self.__common_step(batch, batch_idx)
		self.log('train_CE_LOSS', losses['label_loss'], prog_bar=True)
		self.log('train_L1_LOSS', losses['l1_loss'], prog_bar=True)
		self.log('train_GIoU_LOSS', losses['giou_loss'], prog_bar=True)
		self.log('train_total_loss', loss)

		logs = {
				'train_total_loss': loss,
				'train_CE_LOSS': losses['label_loss'],
				'train_L1_LOSS': losses['l1_loss'],
				'train_GIoU_LOSS': losses['giou_loss']
		}

		batch_dictonary = {
				'loss': loss,
				'log': logs
		}

		return batch_dictonary

	def validation_step(self, batch, batch_idx):
		output, losses, loss = self.__common_step(batch, batch_idx)
		self.log('val_CE_LOSS', losses['label_loss'], on_step=False, on_epoch=True)
		self.log('val_L1_LOSS', losses['l1_loss'], on_step=False, on_epoch=True)
		self.log('val_GIoU_LOSS', losses['giou_loss'], on_step=False, on_epoch=True)
		self.log('val_total_loss', loss, on_step=False)

		logs = {
				'val_total_loss': loss,
				'val_CE_LOSS': losses['label_loss'],
				'val_L1_LOSS': losses['l1_loss'],
				'val_GIoU_LOSS': losses['giou_loss']
		}

		batch_dictonary = {
				'loss': loss,
				'log': logs
		}

		return batch_dictonary

	def predict(self, batch, batch_idx):
		return self.__commong_step(batch, batch_idx)

	def get_optimizre(self):
		opt_dict = {
				'optimizer': self.optimizer
		}