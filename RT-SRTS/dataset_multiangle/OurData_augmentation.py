import torch
import numpy as np
import scipy.ndimage as ndimage

class CT_XRAY_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (ToTensor(), ToTensor(), ToTensor())
    ])

  def __call__(self, img_list):
    return self.augment(img_list)

class multi_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None,Resize_image(size=(1, opt.fine_size, opt.fine_size)),None),
      (None,normalization(),None),
      (ToTensor(), ToTensor(), ToTensor())
    ])

  def __call__(self, img_list):
    return self.augment(img_list)

class List_Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms
  def __call__(self, img_list):
    for t_list in self.transforms:
      if len(t_list) > 1:
        new_img_list = []
        for img, t in zip(img_list, t_list):
          if t is None:
            new_img_list.append(img)
          else:
            new_img_list.append(t(img))
        img_list = new_img_list
      else:
        img_list = t_list[0](img_list)
    return img_list
class Resize_image(object):
  def __init__(self, size=(3,256,256)):
    if not hasattr(size, '__iter__') and hasattr(size, '__len__'):
      raise ValueError('each dimension of size must be defined')
    self.size = np.array(size, dtype=np.float32)
  def __call__(self, img):
    z, x, y = img.shape
    ori_shape = np.array((z, x, y), dtype=np.float32)
    resize_factor = self.size / ori_shape
    img_copy = ndimage.interpolation.zoom(img, resize_factor, order=1)
    return img_copy
class normalization(object):
  def __init__(self,round_v=5):
    self.round_v = round_v
  def __call__(self, image):
    image = image - np.min(image)
    image = np.round(image / np.max(image),self.round_v)
    assert((np.max(image)-1.0 < 1e-3) and (np.min(image) < 1e-3))
    return image
class set_0_1():
  def __init__(self,thresh=0.5):
    self.thresh=thresh
  def __call__(self,img):
    img[img>=self.thresh]=1
    img[img<self.thresh]=0
    return img
class ToTensor(object):
  def __call__(self, img):
    img = torch.from_numpy(img.astype(np.float32))
    return img