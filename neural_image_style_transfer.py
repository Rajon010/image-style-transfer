# Built upon https://github.com/leongatys/PytorchNeuralStyleTransfer

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms
from PIL import Image
from itertools import product

img_dir = 'Images/'
model_dir = 'Models/'

class VGG(nn.Module):
	def __init__(self, pool='max'):
		super(VGG, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		if pool == 'max':
			self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		elif pool == 'avg':
			self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
			
	def forward(self, x, out_keys):
		out = {}
		out['r11'] = F.relu(self.conv1_1(x))
		out['r12'] = F.relu(self.conv1_2(out['r11']))
		out['p1'] = self.pool1(out['r12'])
		out['r21'] = F.relu(self.conv2_1(out['p1']))
		out['r22'] = F.relu(self.conv2_2(out['r21']))
		out['p2'] = self.pool2(out['r22'])
		out['r31'] = F.relu(self.conv3_1(out['p2']))
		out['r32'] = F.relu(self.conv3_2(out['r31']))
		out['r33'] = F.relu(self.conv3_3(out['r32']))
		out['r34'] = F.relu(self.conv3_4(out['r33']))
		out['p3'] = self.pool3(out['r34'])
		out['r41'] = F.relu(self.conv4_1(out['p3']))
		out['r42'] = F.relu(self.conv4_2(out['r41']))
		out['r43'] = F.relu(self.conv4_3(out['r42']))
		out['r44'] = F.relu(self.conv4_4(out['r43']))
		out['p4'] = self.pool4(out['r44'])
		out['r51'] = F.relu(self.conv5_1(out['p4']))
		out['r52'] = F.relu(self.conv5_2(out['r51']))
		out['r53'] = F.relu(self.conv5_3(out['r52']))
		out['r54'] = F.relu(self.conv5_4(out['r53']))
		out['p5'] = self.pool5(out['r54'])
		return [out[key] for key in out_keys]

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, source):
		return source

class MeanVector(nn.Module):
	def __init__(self):
		super(MeanVector, self).__init__()
	def forward(self, source):
		one, n_filter, h, w = source.size()
		m = h * w
		F = source.view(n_filter, m)
		G = torch.mean(F, 1)
		return G

class GramianMatrix(nn.Module):
	def __init__(self, activation_shift):
		super(GramianMatrix, self).__init__()
		self.activation_shift = activation_shift
	def forward(self, source):
		one, n_filter, h, w = source.size()
		m = h * w
		F = source.view(n_filter, m) + self.activation_shift
		G = torch.mm(F, F.transpose(0, 1))
		G.div_(n_filter * m)
		return G

class VarianceVector(nn.Module):
	def __init__(self):
		super(VarianceVector, self).__init__()
	def forward(self, source):
		one, n_filter, h, w = source.size()
		m = h * w
		F = source.view(n_filter, m)
		G = torch.var(F, dim=1)
		G.div_(n_filter)
		return G

class CovarianceMatrix(nn.Module):
	def __init__(self):
		super(CovarianceMatrix, self).__init__()
	def forward(self, source):
		one, n_filter, h, w = source.size()
		m = h * w
		F = source.view(n_filter, m)
		A = torch.mean(F, dim=1).view(-1, 1)
		G = torch.mm(F, F.transpose(0, 1)).div(m) - torch.mm(A, A.transpose(0, 1))
		G.div_(n_filter)
		return G

class LayerLoss(nn.Module):
	def __init__(self, description, raw_target, activation_shift=0.0):
		super(LayerLoss, self).__init__()
		if description == 'raw':
			self.class_, self.arg_tpl = Identity, tuple()
		if description == 'mean':
			self.class_, self.arg_tpl = MeanVector, tuple()
		elif description == 'gramian':
			self.class_, self.arg_tpl = GramianMatrix, (activation_shift,)
		elif description == 'variance':
			self.class_, self.arg_tpl = VarianceVector, tuple()
		elif description == 'covariance':
			self.class_, self.arg_tpl = CovarianceMatrix, tuple()
		self.target = self.class_(*self.arg_tpl)(raw_target).detach()

	def forward(self, source):
		out = nn.MSELoss()(
			self.class_(*self.arg_tpl)(source), 
			self.target,
		)
		return out

img_size = 512
prep = transforms.Compose([
	transforms.Scale(img_size),
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), # turn to BGR
	transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]), # subtract imagenet mean
	transforms.Lambda(lambda x: x.mul_(255.0)), 
])
postpa = transforms.Compose([
	transforms.Lambda(lambda x: x.mul_(1.0/255.0)),
	transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]), # add imagenet mean
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), # turn to RGB
])
postpb = transforms.Compose([
	transforms.ToPILImage(),
])
def postp(tensor):
	return postpb(postpa(tensor).clamp(0.0, 1.0))

vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
	param.requires_grad = False
if torch.cuda.is_available():
	vgg.cuda()

style_layers = ['r11', 'r21', 'r31', 'r41', 'r51'] 
content_layers = ['r42']
loss_layers = style_layers + content_layers
style_weights = [1e3] * len(style_layers)
content_weights = [1e0]
weights = style_weights + content_weights

content_img_name_list = ['beethoven', 'church', 'fate']
style_img_name_list = ['starry_night', 'face', 'ice']
style_description_list = ['mean', 'gramian', 'variance', 'covariance']
style_description_2_activation_shift_list = {
	'mean': [0.0], 
	'gramian': [float(i) for i in range(-600, 700, 100)], 
	'variance': [0.0], 
	'covariance': [0.0],
}

for content_img_name, style_img_name, style_description in product(content_img_name_list, style_img_name_list, style_description_list):
	for activation_shift in style_description_2_activation_shift_list[style_description]:
		print(content_img_name, style_img_name, style_description, activation_shift)

		img_dirs = [img_dir, img_dir]
		img_names = [style_img_name, content_img_name]
		imgs = [Image.open(img_dirs[i] + name + '.jpg').convert('RGB') for i, name in enumerate(img_names)]
		imgs_torch = [prep(img) for img in imgs]
		if torch.cuda.is_available():
			imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
		else:
			imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
		style_img, content_img = imgs_torch

		style_targets = [A.detach() for A in vgg(style_img, style_layers)]
		content_targets = [A.detach() for A in vgg(content_img, content_layers)]

		opt_img = Variable(content_img.data.clone(), requires_grad=True)

		content_desctiprion = 'raw'
		loss_fns = [
			LayerLoss(style_description, raw_target, activation_shift) 
		for raw_target in style_targets] + [
			LayerLoss(content_desctiprion, raw_target) 
		for raw_target in content_targets]
		if torch.cuda.is_available():
			loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

		style_targets, content_targets = None, None

		optimizer = optim.LBFGS([opt_img])
		max_iter = 500
		show_iter = 20
		n_iter = 0

		def closure():
			optimizer.zero_grad()
			out = vgg(opt_img, loss_layers)
			layer_losses = [weights[a] * loss_fns[a](A) for a, A in enumerate(out)]
			loss = sum(layer_losses)
			loss.backward()
			global n_iter
			if n_iter % show_iter == 0:
				print('Iteration:', n_iter, ' loss:', loss.data[0])
			n_iter += 1
			return loss

		while n_iter < max_iter:
			optimizer.step(closure)
			
		out_img = postp(opt_img.data[0].cpu().squeeze())
		out_img.save(content_img_name + '_' + style_img_name + '_' + style_description + '_' + str(activation_shift) + '.jpg')
