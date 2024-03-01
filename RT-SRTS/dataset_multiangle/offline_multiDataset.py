import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from dataset_multiangle.OurData_augmentation import multi_Data_Augmentation
import xml.etree.ElementTree as ET
import random 

class offline_multiDataSet(Dataset):
  def __init__(self, opt, datasetfile):
    super(offline_multiDataSet, self).__init__()
    self.exp = opt.exp
    self.dataroot = opt.dataroot  
    with open(datasetfile, 'r') as f:
        content = f.readlines()
        self.dataset_paths=[i.strip() for i in content]
    self.dataset_size = len(self.dataset_paths)
    self.data_augmentation = multi_Data_Augmentation(opt)
  def __len__(self):
    return self.dataset_size
  

  def __getitem__(self, item):
    file_path = os.path.join(self.dataroot, self.dataset_paths[item], 'ct_xray12_label.h5')
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    tumor_label = np.asarray(hdf5['tumor'])
    backg = 1.0 - tumor_label
    label=np.zeros([2,128,128,128],dtype=float)
    label[0,:,:,:]=tumor_label
    label[1,:,:,:]=backg
    hdf5.close()

    #random angle DRR
    rtkct_filepath=os.path.join('/hdd2/zmx/'+self.exp+'/', 'ct_rtk', str(self.dataset_paths[item])+'.mha')
    geometry_filepath=os.path.join('/hdd2/zmx/1720geometry', '1720geometry.xml')
  

    drr_filepath=os.path.join(self.dataroot, self.dataset_paths[item], 'drr.mhd')
    angle1=0
    angle2=0
    random.seed(3407)
    angle1_list = random.sample(range(0,1080),1000)
    angle2_list = random.sample(range(0,1080),1000)
    angle1 = angle1_list[item]
    angle2 = angle2_list[item]

    first_angle=angle1 if angle1<angle2 else angle2
    arc=angle1 if angle1>angle2 else angle2
    os.system(f'/home/zmx/RTK/rtksimulatedgeometry -f {str(first_angle)} -n 1 --arc {str(arc)} -o {geometry_filepath} && /home/zmx/RTK/rtkforwardprojections -g {geometry_filepath} -i {rtkct_filepath} -o {drr_filepath} --dimension=128 --fp=CudaRayCast')
    drr_img=sitk.ReadImage(drr_filepath)
    drr_array=sitk.GetArrayFromImage(drr_img)

    ct, drr,label = self.data_augmentation([ct_data,drr_array,label])
    return drr,ct,label,self.dataset_paths[item]
