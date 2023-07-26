import numpy as np
import os.path as osp
import torch.nn as nn
import torch
import random
from torch.nn import init
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage import measure
import numpy.linalg as linalg
import SimpleITK as sitk

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network parameters with %s' % init_type)
    net.apply(init_func)

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):#pool_size:50
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, pred, gt):
        epsilon = 1e-5
        assert pred.size(1) == gt.size(1)
        class_num = pred.size(1)
        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(pred[:, i, :, :, :] * gt[:, i, :, :, :])
            union = torch.sum(pred[:, i, :, :, :]) + torch.sum(gt[:, i, :, :, :])
            dice = (2. * inter+epsilon) / (union+epsilon)
            dice_sum += dice
        return 1 - dice_sum / class_num


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        if use_lsgan:
            self.loss = nn.MSELoss()
            print('GAN loss: {}'.format('LSGAN'))
        else:
            self.loss = nn.BCELoss()
            print('GAN loss: {}'.format('Normal'))

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_tensor is None) or
                            (self.real_label_tensor.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.real_label)
                self.real_label_tensor = real_tensor.to(input)
            target_tensor = self.real_label_tensor
        else:
            create_label = ((self.fake_label_tensor is None) or
                            (self.fake_label_tensor.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.fake_label)
                self.fake_label_tensor = fake_tensor.to(input)
            target_tensor = self.fake_label_tensor
        return target_tensor

    def forward(self, input, target_is_real):
        # for multi_scale_discriminator
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        # for patch_discriminator
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
def tensor_back_to_unMinMax(input_image, min, max):
  image = input_image * (max - min) + min
  return image

def refine_label(label):
  label[label>0.5]=1
  label[label<=0.5]=0
  return label

def MAE(arr1, arr2, size_average=True):
  '''
  :param arr1:
    Format-[NDHW], OriImage
  :param arr2:
    Format-[NDHW], ComparedImage
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  if size_average:
    return np.abs(arr1 - arr2).mean()
  else:
    return np.abs(arr1 - arr2).mean(1).mean(1).mean(1)

def MSE(arr1, arr2, size_average=True):
  '''
  :param arr1:
    Format-[NDHW], OriImage
  :param arr2:
    Format-[NDHW], ComparedImage
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  if size_average:
    return np.power(arr1 - arr2, 2).mean()
  else:
    return np.power(arr1 - arr2, 2).mean(1).mean(1).mean(1)

def dice(pred, gt, class_index):
    epsilon = 1e-5
    inter = np.sum(pred[:, class_index, :, :, :] * gt[:, class_index, :, :, :])
    union = np.sum(pred[:, class_index, :, :, :]) + np.sum(gt[:, class_index, :, :, :])
    dice = (2. * inter+epsilon) / (union+epsilon)
    return dice

def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
  zero_mse = np.where(mse == 0)
  mse[zero_mse] = eps
  psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
  # #zero mse, return 100
  psnr[zero_mse] = 100

  if size_average:
    return psnr.mean()
  else:
    return psnr

def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  # Depth
  mse_d = se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
  zero_mse = np.where(mse_d==0)
  mse_d[zero_mse] = eps
  psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
  # #zero mse, return 100
  psnr_d[zero_mse] = 100
  psnr_d = psnr_d.mean(1)

  # Height
  mse_h = se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
  zero_mse = np.where(mse_h == 0)
  mse_h[zero_mse] = eps
  psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
  # #zero mse, return 100
  psnr_h[zero_mse] = 100
  psnr_h = psnr_h.mean(1)

  # Width
  mse_w = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
  zero_mse = np.where(mse_w == 0)
  mse_w[zero_mse] = eps
  psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
  # #zero mse, return 100
  psnr_w[zero_mse] = 100
  psnr_w = psnr_w.mean(1)

  psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
  if size_average:
    return [psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()]
  else:
    return [psnr_d, psnr_h, psnr_w, psnr_avg]

def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)

  N = arr1.shape[0]
  # Depth
  arr1_d = np.transpose(arr1, (0, 2, 3, 1))
  arr2_d = np.transpose(arr2, (0, 2, 3, 1))
  ssim_d = []
  for i in range(N):
    ssim = measure.compare_ssim(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, multichannel=True)
    ssim_d.append(ssim)
  ssim_d = np.asarray(ssim_d, dtype=np.float64)

  # Height
  arr1_h = np.transpose(arr1, (0, 1, 3, 2))
  arr2_h = np.transpose(arr2, (0, 1, 3, 2))
  ssim_h = []
  for i in range(N):
    ssim = measure.compare_ssim(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, multichannel=True)
    ssim_h.append(ssim)
  ssim_h = np.asarray(ssim_h, dtype=np.float64)

  # Width
  # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
  # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
  ssim_w = []
  for i in range(N):
    ssim = measure.compare_ssim(arr1[i], arr2[i], data_range=PIXEL_MAX, multichannel=True)
    ssim_w.append(ssim)
  ssim_w = np.asarray(ssim_w, dtype=np.float64)

  ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

  if size_average:
    return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
  else:
    return [ssim_d, ssim_h, ssim_w, ssim_avg]


def getPred(data, idx, test_idx=None):
    pred = data[idx, ...]
    return pred

def getGroundtruth(data, idx, test_idx=None):
    groundtruth = data[idx, ...]
    return groundtruth

def getErrorMetrics(im_pred, im_gt, mask=None):
    im_pred = np.array(im_pred).astype(np.float64)
    im_gt = np.array(im_gt).astype(np.float64)
    # sanity check
    assert(im_pred.flatten().shape==im_gt.flatten().shape)
    # RMSE
    rmse_pred = measure.compare_nrmse(im_true=im_gt, im_test=im_pred)
    # PSNR
    psnr_pred = measure.compare_psnr(im_true=im_gt, im_test=im_pred)
    # SSIM
    ssim_pred = measure.compare_ssim(X=im_gt, Y=im_pred)
    # MSE
    mse_pred = mean_squared_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    # MAE
    mae_pred = mean_absolute_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    return mae_pred, mse_pred, rmse_pred, psnr_pred, ssim_pred

def save_mhd(image_array,outputpath):
  image=sitk.GetImageFromArray(image_array)
  sitk.WriteImage(image,outputpath)
  
def calculate_angle_accuracy(pred_angle,gt):
    errors=[]
    nums=len(pred_angle)
    for i in range(len(pred_angle)):
        errors.append(abs(pred_angle[i]-gt[i]))
    error05=0
    error10=0
    error15=0
    error20=0
    error25=0
    error30=0
    error35=0
    error40=0
    error45=0
    error50=0
    error55=0
    error60=0
    for error in errors:
        if error<=0.5:
            error05+=1
        if error<=1.0:
            error10+=1
        if error<=1.5:
            error15+=1
        if error<=2.0:
            error20+=1
        if error<=2.5:
            error25+=1
        if error<=3.0:
            error30+=1
        if error<=3.5:
            error35+=1
        if error<=4.0:
            error40+=1
        if error<=4.5:
            error45+=1
        if error<=5.0:
            error50+=1
        if error<=5.5:
            error55+=1
        if error<=6.0:
            error60+=1
    print('允许0.5范围角度预测差的准确值：',error05/nums)
    print('允许1.0范围角度预测差的准确值：',error10/nums)
    print('允许1.5范围角度预测差的准确值：',error15/nums)
    print('允许2.0范围角度预测差的准确值：',error20/nums)
    print('允许2.5范围角度预测差的准确值：',error25/nums)
    print('允许3.0范围角度预测差的准确值：',error30/nums)
    print('允许3.5范围角度预测差的准确值：',error35/nums)
    print('允许4.0范围角度预测差的准确值：',error40/nums)
    print('允许4.5范围角度预测差的准确值：',error45/nums)
    print('允许5.0范围角度预测差的准确值：',error50/nums)
    print('允许5.5范围角度预测差的准确值：',error55/nums)
    print('允许6.0范围角度预测差的准确值：',error60/nums)
    return errors

