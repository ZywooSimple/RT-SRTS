import torch.nn as nn
import torch
import math
from torch.nn import init

# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes,
			kernel_size=1, stride=stride, padding=0, 
			bias=False)

def conv2x2(in_planes, out_planes, stride=2):
	return nn.Conv2d(in_planes, out_planes,
			kernel_size=2, stride=stride, padding=0, 
			bias=False)

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes,
			kernel_size=3, stride=stride, padding=1, 
			bias=False)

def conv4x4(in_planes, out_planes, stride=2):
	return nn.Conv2d(in_planes, out_planes,
			kernel_size=4, stride=stride, padding=1, 
			bias=False)
# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=1, stride=stride, padding=0, 
			bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=3, stride=stride, padding=1, 
			bias=False)

def conv4x4x4(in_planes, out_planes, stride=2):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=4, stride=stride, padding=1, 
			bias=False)
# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=1, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv2x2(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=2, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv3x3(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=3, stride=stride, padding=1, output_padding=0, 
			bias=False)

def deconv4x4(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=4, stride=stride, padding=1, output_padding=0, 
			bias=False)

# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=1, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv3x3x3(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=3, stride=stride, padding=1, output_padding=0, 
			bias=False)

def deconv4x4x4(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=4, stride=stride, padding=1, output_padding=0, 
			bias=False)
class Dimension_UpsampleCutBlock(nn.Module):   #connectionB跳层连接
  def __init__(self, input_channel, output_channel,activation=nn.ReLU(True), use_bias=True):
    super(Dimension_UpsampleCutBlock, self).__init__()
    self.output_channel = output_channel
    basic2d_block = [
      nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0, bias=use_bias),
      nn.InstanceNorm2d(output_channel),
      activation
    ]  #二维卷积使二维特征图通道数为decoder对应
    self.basic2d_block = nn.Sequential(*basic2d_block)
    self.softmax=nn.Softmax(dim=2)
    basic3d_block = [
      nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
      nn.InstanceNorm3d(output_channel),
      activation
    ]
    self.basic3d_block = nn.Sequential(*basic3d_block)
  def forward(self, input):
    # input's shape is [NCHW]
    N,_,H,W = input.size()
    temp_c=self.basic2d_block(input)
    c_hw=temp_c.reshape((N,self.output_channel,H*W))  #channel attention add in 2D convolution
    hw_c=temp_c.reshape((N,H*W,self.output_channel))
    c_c=self.softmax(torch.matmul(c_hw,hw_c))
    attention_2D=torch.matmul(c_c,c_hw).reshape((N,self.output_channel,H,W))
    return self.basic3d_block(attention_2D.unsqueeze(2).expand(N,self.output_channel,H,H,W))

def _make_layers(in_channels, output_channels, type, batch_norm=False, activation=None):
	layers = []

	if type == 'conv1_s1':
		layers.append(conv1x1(in_channels, output_channels, stride=1))
	elif type == 'conv2_s2':
		layers.append(conv2x2(in_channels, output_channels, stride=2))
	elif type == 'conv3_s1':
		layers.append(conv3x3(in_channels, output_channels, stride=1))
	elif type == 'conv4_s2':
		layers.append(conv4x4(in_channels, output_channels, stride=2))
	elif type == 'deconv1_s1':
		layers.append(deconv1x1(in_channels, output_channels, stride=1))
	elif type == 'deconv2_s2':
		layers.append(deconv2x2(in_channels, output_channels, stride=2))
	elif type == 'deconv3_s1':
		layers.append(deconv3x3(in_channels, output_channels, stride=1))
	elif type == 'deconv4_s2':
		layers.append(deconv4x4(in_channels, output_channels, stride=2))
	elif type == 'conv1x1_s1':
		layers.append(conv1x1x1(in_channels, output_channels, stride=1))
	elif type == 'deconv1x1_s1':
		layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
	elif type == 'deconv3x3_s1':
		layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
	elif type == 'deconv4x4_s2':
		layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
	elif type == 'skip_conection':
		layers.append(Dimension_UpsampleCutBlock(in_channels, output_channels))

	else:
		raise NotImplementedError('layer type [{}] is not implemented'.format(type))

	if batch_norm == '2d':
		layers.append(nn.InstanceNorm2d(output_channels))
	elif batch_norm == '3d':
		layers.append(nn.InstanceNorm3d(output_channels))

	if activation == 'relu':
		layers.append(nn.ReLU(inplace=True))
	elif activation == 'sigm':
		layers.append(nn.Sigmoid())
	elif activation == 'leakyrelu':
		layers.append(nn.LeakyReLU(-1.0, False))
	else:
		if activation is not None:
			raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

	return nn.Sequential(*layers)


def _initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.InstanceNorm3d):
			# m.weight.data.fill_(1)
			# m.bias.data.zero_()
			pass
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.01)
			m.bias.data.zero_()


class ReconNet(nn.Module):

	def __init__(self, in_channels, out_channels, gain=0.02, init_type='standard'):
		super(ReconNet, self).__init__()

		######### representation network - convolution layers
		self.conv_layer1 = _make_layers(in_channels, 	256, 	'conv4_s2', False)
		self.conv_layer2 = _make_layers(256, 		256, 	'conv3_s1', '2d')
		self.relu2 = nn.ReLU(inplace=True)
		self.conv_layer3 = _make_layers(256, 		512, 	'conv4_s2', '2d', 'relu')
		self.conv_layer4 = _make_layers(512, 		512, 	'conv3_s1', '2d')
		self.relu4 = nn.ReLU(inplace=True)
		self.conv_layer5 = _make_layers(512, 		1024, 	'conv4_s2', '2d', 'relu')
		self.conv_layer6 = _make_layers(1024, 		1024, 	'conv3_s1', '2d')
		self.relu6 = nn.ReLU(inplace=True)
		self.conv_layer7 = _make_layers(1024, 		2048, 	'conv4_s2', '2d', 'relu')
		self.conv_layer8 = _make_layers(2048, 		2048, 	'conv3_s1', '2d')
		self.relu8 = nn.ReLU(inplace=True)
		self.conv_layer9 = _make_layers(2048, 		4096, 	'conv4_s2', '2d', 'relu')
		self.conv_layer10 = _make_layers(4096, 		4096, 	'conv3_s1', '2d')
		self.relu10 = nn.ReLU(inplace=True)


		######### transform module
		self.trans_layer1 = _make_layers(4096, 		4096, 	'conv1_s1', False, 'relu')
		self.trans_layer2 = _make_layers(1024, 		1024, 	'deconv1x1_s1', False, 'relu')

		######### skip-connection structure----reconstruction
		self.skip_connection5=_make_layers(4096,	1024,	'skip_conection')
		self.skip_connection4=_make_layers(2048,	512,	'skip_conection')
		self.skip_connection3=_make_layers(1024,	256,	'skip_conection')
		self.skip_connection2=_make_layers(512,		128,	'skip_conection')
		self.skip_connection1=_make_layers(256,		64,	'skip_conection')

		######### generation network - deconvolution layers
		self.deconv_layer10 = _make_layers(2048, 	512, 	'deconv4x4_s2', '3d', 'relu')
		self.deconv_layer8 = _make_layers(1024,		256, 	'deconv4x4_s2', '3d', 'relu')
		self.deconv_layer7 = _make_layers(256,		256, 	'deconv3x3_s1', '3d', 'relu')
		self.deconv_layer6 = _make_layers(512,		128, 	'deconv4x4_s2', '3d', 'relu')
		self.deconv_layer5 = _make_layers(128,		128, 	'deconv3x3_s1', '3d', 'relu')
		self.deconv_layer4 = _make_layers(256,		64, 	'deconv4x4_s2', '3d', 'relu')
		self.deconv_layer3 = _make_layers(64,		64, 	'deconv3x3_s1', '3d', 'relu')
		self.deconv_layer2 = _make_layers(128,		32, 	'deconv4x4_s2', '3d', 'relu')
		self.deconv_layer1 = _make_layers(32,		32, 	'deconv3x3_s1', '3d', 'relu')
		self.deconv_layer0 = _make_layers(32,		1, 		'conv1x1_s1', False, 'relu')
		self.output_layer = _make_layers(128,	out_channels, 'conv1_s1', False, 'leakyrelu')#最后一层leakyRelu

		######### segmentation network 
		self.s_deconv_layer10 = _make_layers(2048, 	512, 	'deconv4x4_s2', '3d', 'relu')
		self.s_deconv_layer8 = _make_layers(1024,		256, 	'deconv4x4_s2', '3d', 'relu')
		self.s_deconv_layer7 = _make_layers(256,		256, 	'deconv3x3_s1', '3d', 'relu')
		self.s_deconv_layer6 = _make_layers(512,		128, 	'deconv4x4_s2', '3d', 'relu')
		self.s_deconv_layer5 = _make_layers(128,		128, 	'deconv3x3_s1', '3d', 'relu')
		self.s_deconv_layer4 = _make_layers(256,		64, 	'deconv4x4_s2', '3d', 'relu')
		self.s_deconv_layer3 = _make_layers(64,		64, 	'deconv3x3_s1', '3d', 'relu')
		self.s_deconv_layer2 = _make_layers(128,		32, 	'deconv4x4_s2', '3d', 'relu')
		self.s_deconv_layer1 = _make_layers(32,		32, 	'deconv3x3_s1', '3d', 'relu')
		self.s_deconv_layer0_1 = _make_layers(32,		2, 		'conv1x1_s1', False, 'relu')
		self.s_deconv_layer0_2 = _make_layers(32,		2, 		'conv1x1_s1', False, 'relu')
		self.seg_output_layer_1 = nn.Softmax(dim=1)
		self.seg_output_layer_2 = nn.Softmax(dim=1)

		
		# network initialization
		_initialize_weights(self)


	def forward(self, x, out_feature=False):
		##################################################################################################### representation network
		#x:[1,1,128,128]
		conv1 = self.conv_layer1(x)
		conv2 = self.conv_layer2(conv1) 
		relu2 = self.relu2(conv1 + conv2)#[256,64,64]
		skip_features1=self.skip_connection1(relu2) #[64,64,64,64]

		conv3 = self.conv_layer3(relu2)
		conv4 = self.conv_layer4(conv3)
		relu4 = self.relu4(conv3 + conv4)#[512,32,32]
		skip_features2=self.skip_connection2(relu4) #[128,32,32,32]

		conv5 = self.conv_layer5(relu4)
		conv6 = self.conv_layer6(conv5)
		relu6 = self.relu6(conv5 + conv6)#[1024,16,16]
		skip_features3=self.skip_connection3(relu6) #[256,16,16,16]

		conv7 = self.conv_layer7(relu6)
		conv8 = self.conv_layer8(conv7)
		relu8 = self.relu8(conv7 + conv8)#[2048,8,8]
		skip_features4=self.skip_connection4(relu8) #[512,8,8,8]

		conv9 = self.conv_layer9(relu8)
		conv10 = self.conv_layer10(conv9)
		relu10 = self.relu10(conv9 + conv10)#[4096,4,4]
		skip_features5=self.skip_connection5(relu10) #[1024,4,4,4]

		##################################################################################################### transform module
		features = self.trans_layer1(relu10)#[4096,4,4]
		trans_features = features.view(-1, 1024, 4, 4, 4)
		trans_features = self.trans_layer2(trans_features) #[1024,4,4,4]

		##################################################################################################### generation network
		deconv10 = self.deconv_layer10(torch.cat((trans_features,skip_features5),dim=1))
		deconv8 = self.deconv_layer8(torch.cat((deconv10,skip_features4),dim=1))
		deconv7 = self.deconv_layer7(deconv8)
		deconv6 = self.deconv_layer6(torch.cat((deconv7,skip_features3),dim=1))
		deconv5 = self.deconv_layer5(deconv6)
		deconv4 = self.deconv_layer4(torch.cat((deconv5,skip_features2),dim=1))
		deconv3 = self.deconv_layer3(deconv4)
		deconv2 = self.deconv_layer2(torch.cat((deconv3,skip_features1),dim=1))
		deconv1 = self.deconv_layer1(deconv2)

		##################################################################################################### segmentation network
		s_deconv10 = self.s_deconv_layer10(torch.cat((trans_features,skip_features5),dim=1))
		s_deconv8 = self.s_deconv_layer8(torch.cat((s_deconv10,skip_features4),dim=1))
		s_deconv7 = self.s_deconv_layer7(s_deconv8)
		s_deconv6 = self.s_deconv_layer6(torch.cat((s_deconv7,skip_features3),dim=1))
		s_deconv5 = self.s_deconv_layer5(s_deconv6)
		s_deconv4 = self.s_deconv_layer4(torch.cat((s_deconv5,skip_features2),dim=1))
		s_deconv3 = self.s_deconv_layer3(s_deconv4)
		s_deconv2 = self.s_deconv_layer2(torch.cat((s_deconv3,skip_features1),dim=1))
		s_deconv1 = self.s_deconv_layer1(s_deconv2)
		s_deconv0_1 = self.s_deconv_layer0_1(s_deconv1)#第一次输出卷积
		score=self.seg_output_layer_1(s_deconv0_1)
		score_top, _ = score.topk(k=2, dim=1)
		uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)
		uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)
		confidence_map = 1-uncertainty
		s_deconv0_2=self.s_deconv_layer0_2(torch.mul(s_deconv1,confidence_map))
		seg_out=self.seg_output_layer_2(s_deconv0_2)
		### reconstruction output
		out = self.deconv_layer0(deconv1)
		out = torch.squeeze(out, 1)
		out = self.output_layer(out)

		if out_feature:
			return out, features, trans_features
		else:
			return out,seg_out

