import numpy as np
import SimpleITK as sitk
import os
import shutil


def return_mask(a,b):
    map1=np.zeros(a.shape,dtype=int)
    map1[a==True]=1
    map1[a==False]=0
    map2=np.zeros(b.shape,dtype=int)
    map2[b==True]=1
    map2[b==False]=0
    return (map1+map2)==2
def dcm2rtked_array(inputfile,outfile):
    img = sitk.ReadImage(inputfile)
    img_array = sitk.GetArrayFromImage(img)
    HU = img_array
    ED = np.zeros(HU.shape, HU.dtype)
    ED[HU <= -966] = 0
    ED[return_mask((-966 < HU), (HU <= -172))] = (HU[return_mask((-966 < HU), (HU <= -172))] + 966) / (-172 + 966) * (
                0.853 - 0) + 0.0
    ED[return_mask((-172 < HU), (HU <= -83))] = (HU[return_mask((-172 < HU), (HU <= -83))] + 172) / (-83 + 172) * (
                0.944 - 0.853) + 0.853
    ED[return_mask((-83 < HU), (HU <= 2.7))] = (HU[return_mask((-83 < HU), (HU <= 2.7))] + 83) / (2.7 + 83) * (
                1 - 0.944) + 0.944
    ED[return_mask((2.7 < HU), (HU <= 124.0))] = (HU[return_mask((2.7 < HU), (HU <= 124.0))] - 2.7) / (124 - 2.7) * (
                1.146 - 1.000) + 1.000
    ED[return_mask((124.0 < HU), (HU <= 340.0))] = (HU[return_mask((124.0 < HU), (HU <= 340.0))] - 124.0) / (
                340.0 - 124.0) * (1.319 - 1.146) + 1.146
    ED[return_mask((340.0 < HU), (HU <= 924.3))] = (HU[return_mask((340.0 < HU), (HU <= 924.3))] - 340.0) / (
                924.3 - 340.0) * (1.867 - 1.319) + 1.319
    ED[return_mask((924.3 < HU), (HU <= 3071.0))] = (HU[return_mask((924.3 < HU), (HU <= 3071.0))] - 924.3) / (
                3071.0 - 924.3) * (3.6 - 1.867) + 1.867
    ED[return_mask((3071.0 < HU), (HU <= 6000.0))] = (HU[return_mask((3071.0 < HU), (HU <= 6000.0))] - 3071.0) / (
                6000.0 - 3071.0) * (6.0 - 3.6) + 3.6
    ED[return_mask((6000.0 < HU), (HU <= 15000.0))] = (HU[return_mask((6000.0 < HU), (HU <= 15000.0))] - 6000.0) / (
                15000.0 - 6000.0) * (12.0 - 6.0) + 6.0
    img2=sitk.GetImageFromArray(ED)
    sitk.WriteImage(img2, outfile)
    return 0
# for i in range(1,1081):
#     inputfile=os.path.join('/hdd/1/zmx/CT128_rtked',str(i)+'_rtked.mha')
#     image=sitk.ReadImage(inputfile)
#     image.SetOrigin((-64.0,-64.0,-64.0))
#     sitk.WriteImage(image,inputfile)
#     print(str(i))
def lv_bo(image_array):
    copy_array = image_array
    temp = np.squeeze(copy_array, 0)
    # 3*3的均值滤波,filter_size必须是大于3的奇数
    filter_size = 3
    filter_img = np.ones((filter_size, filter_size))
    # 3*3的锐化,注释掉的那部分
    # filter_sharpe = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])


    pad_num = int((filter_size - 1) / 2)
    img_pad = np.pad(temp, (pad_num, pad_num), mode="edge")
    result = np.copy(img_pad)
    m, n = img_pad.shape
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            result[i, j] = np.sum((filter_img * img_pad[i-pad_num:i+pad_num+1, j-pad_num:j+pad_num+1])/(filter_size**2))
    result_array = result[pad_num:m - pad_num, pad_num:n - pad_num]
    output = np.expand_dims(result_array,0)
    return output
if __name__=='__main__':
    for i in range(1,1081):
        if i==699:
            continue
        inputfile=os.path.join('/hdd2/zmx/Case10Pack_preprocess/h5py/',str(i),'drr.mhd')
        image=sitk.ReadImage(inputfile)
        img_array = sitk.GetArrayFromImage(image)

        gaosi_noise = np.random.normal(0, 1, img_array.shape)
        output = img_array + gaosi_noise
        output = lv_bo(output)
        output = sitk.GetImageFromArray(output)
        outfilepath=os.path.join('/hdd2/zmx/Case10Pack_preprocess/zhu_drr/',str(i)+'.mha')
        sitk.WriteImage(output,outfilepath)
        print(i)