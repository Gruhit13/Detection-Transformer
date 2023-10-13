import torch
from torch import nn
import torch.nn.functional as F

def get_activation_fn(activation):
  """Returns the corresponding activation function based on the argument"""

  if activation == "relu":
	return F.relu
  elif activation == "gelu":
	return F.gelu
  elif activation == "glu":
	return F.glu
  else:
	raise ValueError("Invalid value for argument passed. No such activation obtained.")


# A simple encoder block that will be stacked for creating transfer
class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, activation='relu'):
	super(EncoderBlock, self).__init__()

	self.self_attn = nn.MultiheadAttention(d_model, num_heads)
	self.linear1 = nn.Linear(d_model, ff_dim)
	self.dropout = nn.Dropout(dropout)
	self.linear2 = nn.Linear(ff_dim, d_model)

	self.norm1 = nn.LayerNorm(d_model)
	self.norm2 = nn.LayerNorm(d_model)
	self.dropout1 = nn.Dropout(dropout)
	self.dropout2 = nn.Dropout(dropout)

	self.activation  = get_activation_fn(activation)

  def with_pos_embed(self, x, pos):
	return x+pos if pos is not None else x

  def forward(
	  self,
	  src,
	  src_mask = None,
	  src_key_padding_mask = None,
	  pos = None
  ):

	q = k = self.with_pos_embed(src, pos)
	src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
						  key_padding_mask=src_key_padding_mask)[0]

	src = src + self.dropout1(src2)
	src = self.norm1(src)
	src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
	src = src + self.dropout2(src2)
	src = self.norm2(src)

	return src


# Decoder block to stack up for creating transformer
class DecoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, ff_dim, dropout=0.1,
			   activation='relu'):

	super(DecoderBlock, self).__init__()
	self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
	self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

	self.linear1 = nn.Linear(d_model, ff_dim)
	self.dropout = nn.Dropout(dropout)
	self.linear2 = nn.Linear(ff_dim, d_model)

	self.norm1 = nn.LayerNorm(d_model)
	self.norm2 = nn.LayerNorm(d_model)
	self.norm3 = nn.LayerNorm(d_model)
	self.dropout1 = nn.Dropout(dropout)
	self.dropout2 = nn.Dropout(dropout)
	self.dropout3 = nn.Dropout(dropout)

	self.activation = get_activation_fn(activation)

  def with_pos_embed(self, x, pos):
	return x+pos if pos is not None else x

  def forward(self, tgt, memory,
			  tgt_mask=None,
			  memory_mask=None,
			  tgt_key_padding_mask=None,
			  memory_key_padding_mask=None,
			  pos=None,
			  query_pos=None):

	  q = k = self.with_pos_embed(tgt, query_pos)
	  tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
							key_padding_mask=tgt_key_padding_mask)[0]

	  tgt = tgt + self.dropout1(tgt2)
	  tgt = self.norm1(tgt)

	  tgt2 = self.cross_attn(query=q, key=self.with_pos_embed(memory, pos), value=memory,
							 attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
	  tgt = tgt + self.dropout2(tgt2)
	  tgt = self.norm2(tgt)
	  tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
	  tgt = tgt + self.dropout3(tgt2)
	  tgt = self.norm3(tgt)
	  return tgt

class Transformer(nn.Module):
  def __init__(self, d_model, num_heads, ff_dim, n_layers, dropout=0.1, activation='relu'):
	super(Transformer, self).__init__()

	self.encoder_layers = nn.ModuleList([
		EncoderBlock(d_model, num_heads, ff_dim, dropout, activation) for _ in range(n_layers)
	])

	self.decoder_layers = nn.ModuleList([
		DecoderBlock(d_model, num_heads, ff_dim, dropout, activation) for _ in range(n_layers)
	])

	self.d_model = d_model
	self.n_heads = num_heads

	self._reset_parameters()

  def _reset_parameters(self):
	for p in self.parameters():
	  if p.dim() > 1:
		nn.init.xavier_uniform_(p)

  def forward(self, src, mask, query_embed, pos_embed):
	# Flatten BxCxHxW => BxCxHW
	B, C, H, W = src.shape
	src = src.flatten(2).permute(2, 0, 1)
	pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
	query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
	mask = mask.flatten(1)

	enc_output = src
	for layer in self.encoder_layers:
	  enc_output = layer(enc_output, src_key_padding_mask=mask, pos=pos_embed)

	tgt = torch.zeros_like(query_embed)

	dec_output = tgt
	for layer in self.decoder_layers:
	  dec_output = layer(dec_output, memory=enc_output,
						 memory_key_padding_mask=mask,
						 pos=pos_embed, query_pos=query_embed)

	dec_output = dec_output.unsqueeze(0)

	return dec_output.transpose(1, 2), enc_output.permute(1, 2, 0).view(B, C, H, W)