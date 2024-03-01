import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from net import ReconNet
from dataset_multiangle.offline_multiDataset import offline_multiDataSet
# from dataset_multiangle.offline_singleDataset import offline_singleDataSet
from dataset_multiangle.OurFn import our_gan
import tensorboardX
import shutil
from utils import *
import time
import math

def parse_args():
  parser = argparse.ArgumentParser(description='testing_script')
  parser.add_argument('--exp', type=str, default='zm_p2', dest='exp',
                     help='exp_name ')
  parser.add_argument('--date', type=str, default='xray_try_zm_seed')
  parser.add_argument('--gpu', type=str, default='0', dest='gpuid',
                    help='gpu is split by ,')
                    
  parser.add_argument('--model_name', type=str, default='best_model',help='best_model or curr_model') 
  parser.add_argument('--batch_size', type=int, default=1, dest='batch_size',
                     help='batch_size')
  parser.add_argument('--CT_MIN_MAX', type=list, default=[0,4096])
  parser.add_argument('--experiment_path', type=str, default='/hdd2/zmx/experiment/')
  parser.add_argument('--phase',type=str,default='test')
  parser.add_argument('--drr_generation',type=int,default=1)
  parser.add_argument('--arch', type=str, default='ReconNet', dest='arch',
                    help='architecture of network')
  parser.add_argument('--seed', type=int, default=1, 
                      metavar='N', help='manual seed for GPUs to generate random numbers')
  parser.add_argument('--num_views', type=int, default=1,
                      help='number of views/projections in inputs')
  parser.add_argument('--output_channel', type=int, default=128,
                      help='dimension of ouput 3D model size')
  parser.add_argument('--output_size', type=int, default=128,
                      help='dimension of ouput 3D model size')
  parser.add_argument('--fine_size', type=int, default=128,
                      help='dimension of ouput 3D model size')
  parser.add_argument('--classes', type=int, default=3, dest='classes',
                      help='output_classes')
  parser.add_argument('--init_gain', type=float, default=0.02, dest='init_gain',
                      help='init_gain')
  parser.add_argument('--init_type', type=str, default='standard', dest='init_type',
                      help='init_type')
  args = parser.parse_args()
  return args

def test(val_loader, model, criterion, criterionDice, args, tb, mode):
    avg_dict = dict()
    test_recon_loss = AverageMeter()
    test_seg_loss=AverageMeter()
    time_=AverageMeter()
    pred = np.zeros((args.test, args.output_channel, args.output_size, args.output_size), dtype=np.float32)
    gt = np.zeros((args.test, args.output_channel, args.output_size, args.output_size), dtype=np.float32)

    
    if not os.path.exists(os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'output')):
      os.makedirs(os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'output'))
    else:
      shutil.rmtree(os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'output'))
      os.makedirs(os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'output'))
    total_image_index=[]
    model.eval()
    with torch.no_grad():
      for i,(input, target,label,image_index) in enumerate(val_loader): #,angle
        # send the data to cuda device
        input = input.to(args.device)
        target = target.to(args.device)
        label = label.to(args.device)
        start_time=time.time()
        output,seg_out= model(input)
        using_time=time.time()-start_time
        time_.update(using_time,1)

        Recon_loss = criterion(output, target)
        Seg_loss=criterionDice(seg_out,label)         
        test_recon_loss.update(Recon_loss.data.item(), input.size(0))
        test_seg_loss.update(Seg_loss.data.item(), input.size(0))

        pred[i, :, :, :] = output.cpu().numpy()
        gt[i, :, :, :] = target.cpu().numpy()
        print('{0}: [{1}/{2}]\t'
              'Recon Loss: {ReconLoss.val:.5f} (average:{ReconLoss.avg:.5f})\t \
              Seg Loss: {SegLoss.val:.5f} (average:{SegLoss.avg:.5f})\t'.format(
               mode, i+1, len(val_loader), 
               ReconLoss = test_recon_loss,SegLoss=test_seg_loss))

        ###############################################
        generate_CT = output.cpu().numpy()
        real_CT = target.cpu().numpy()
        pred_label = seg_out.cpu().numpy()
        label=label.cpu().numpy()
        ###############################################

        #save mhd image
        outmhd_dir=os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'output')
        groud_ct_path=os.path.join(outmhd_dir,image_index[0]+'_CT.mhd')
        #out_ct_path=os.path.join(outmhd_dir,image_index[0]+'_'+angle[0]+'_GeneratedCT.mhd')
        out_ct_path=os.path.join(outmhd_dir,image_index[0]+'_'+'_GeneratedCT.mhd')
        out_seg_path=os.path.join(outmhd_dir,image_index[0]+'_GeneratedSeg'+'.mhd')
        groud_seg_path=os.path.join(outmhd_dir,image_index[0]+'_Seg'+'.mhd')
        save_mhd(np.squeeze(generate_CT),out_ct_path)
        save_mhd(np.squeeze(real_CT),groud_ct_path)
        save_mhd(np.transpose(np.squeeze(label),(1,2,3,0)),groud_seg_path)
        # refine the output geberated label
        pred_label=refine_label(pred_label)
        save_mhd(np.transpose(np.squeeze(pred_label),(1,2,3,0)),out_seg_path) 

        total_image_index.append(image_index[0])
        print(image_index[0]+'has been tested and saved')
        # CT range 0-1
        mae0_1 = MAE(real_CT, generate_CT, size_average=False)
        mse0_1 = MSE(real_CT, generate_CT, size_average=False)
        ssim0_1 = Structural_Similarity(real_CT, generate_CT, size_average=False, PIXEL_MAX=1.0)
        # To HU coordinate CT range 0-4096
        generate_CT_4096 = tensor_back_to_unMinMax(generate_CT, args.CT_MIN_MAX[0], args.CT_MIN_MAX[1]).astype(np.int32)
        real_CT_4096 = tensor_back_to_unMinMax(real_CT, args.CT_MIN_MAX[0], args.CT_MIN_MAX[1]).astype(np.int32)
        psnr_3d_4096 = Peak_Signal_to_Noise_Rate_3D(real_CT_4096, generate_CT_4096, size_average=False, PIXEL_MAX=4096)
        psnr_4096 = Peak_Signal_to_Noise_Rate(real_CT_4096, generate_CT_4096, size_average=False, PIXEL_MAX=4096)
        mae_4096 = MAE(real_CT_4096, generate_CT_4096, size_average=False)
        mse_4096 = MSE(real_CT_4096, generate_CT_4096, size_average=False)
        dice0 = np.asarray([dice(pred_label, label, 0)]) #tumor
        dice1 = np.asarray([dice(pred_label, label, 1)]) #background
        mean_dice = np.asarray((dice0 + dice1) / 2.0)

        metrics_list = [('MAE0-1', mae0_1), ('MSE0-1', mse0_1), ('MAE_4096', mae_4096), ('MSE_4096', mse_4096),
                        ('PSNR_4096', psnr_3d_4096), ('PSNR1_depth_4096', psnr_4096[0]),
                        ('PSNR2_height_4096', psnr_4096[1]), ('PSNR3_width_4096', psnr_4096[2]), ('PSNR_avg_4096', psnr_4096[3]),
                        ('SSIM1_depth_0_1', ssim0_1[0]), ('SSIM2_height_0_1', ssim0_1[1]), ('SSIM3_width_0_1', ssim0_1[2]), ('SSIM_avg_0_1', ssim0_1[3]),
                        ('dice_tumor', dice0),('dice_background', dice1),('dice_mean', mean_dice)]

        for key, value in metrics_list:
            if avg_dict.get(key) is None:
                avg_dict[key] = [] + value.tolist()
            else:
                avg_dict[key].extend(value.tolist())
    print(f'average generation time:{time_.avg}')
    save_txt_path = os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'X2CT_metrics.txt')
    with open(save_txt_path, 'w') as f:
        f.write(f'total images:{len(val_loader)}')
        for key, value in avg_dict.items():
            print('Metric:{}------>{} \n'.format(key, np.round(np.mean(value), 5)))
            log_metrics = 'Metric:{}------>{} \n '.format(key, np.round(np.mean(value), 5))
            f.write(log_metrics)
        f.write(f'average generation time:{time_.avg}')
    
    return test_recon_loss, pred, gt,total_image_index

if __name__ == '__main__':
    #define device
    args = parse_args()
    assert (torch.cuda.is_available())
    split_gpu = str(args.gpuid).split(',')
    args.gpu_ids = [int(i) for i in split_gpu]
    args.device = torch.device('cuda:{}'.format(args.gpu_ids[0]))
    
    # set random seed for GPUs for reproducible
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #define filefolder
    test_datasetfile = args.experiment_path + 'test.txt' 
    args.model_output_path = os.path.join(args.experiment_path ,args.exp ,args.date,'model')
    args.dataroot=os.path.join('/hdd2/zmx',args.exp,'h5py')
    log_dir=os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'test_log')
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    else:
      shutil.rmtree(log_dir)
      os.makedirs(log_dir)
    tb = tensorboardX.SummaryWriter(log_dir=log_dir)
    
    # define model
    model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel, gain=args.init_gain, init_type=args.init_type)
    model.to(args.gpu_ids[0])

    # define loss function
    criterion = nn.MSELoss(reduction='mean').to(args.device)
    criterionDice = DiceMeanLoss().to(args.device)

    # enable CUDNN benchmark
    cudnn.benchmark = True

    # test_dataloader
    test_dataset = offline_multiDataSet(args, test_datasetfile)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=our_gan)
    args.test=len(test_loader)

    # load model
    ckpt_file = os.path.join(args.model_output_path,args.model_name+'.pth.tar')
    if os.path.isfile(ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' ".format(ckpt_file))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_file))
    print('best_training_epoch:',checkpoint['epoch'])

    # test evaluation 
    _,pred_data, gt_data, total_image_index= test(test_loader, model, criterion,criterionDice, args, tb,mode='Test')
  
    # calculate metric of nature
    mae_avg = []
    mse_avg = []
    rmse_avg = [] 
    psnr_avg = []
    ssim_avg = []
    for idx in range(args.test):
        pred = getPred(pred_data, idx)
        groundtruth = getGroundtruth(gt_data, idx)
        mae_pred, mse_pred, rmse_pred, psnr_pred, ssim_pred = getErrorMetrics(im_pred=pred, im_gt=groundtruth)
        mae_avg.append(mae_pred)
        mse_avg.append(mse_pred)
        rmse_avg.append(rmse_pred)
        psnr_avg.append(psnr_pred)
        ssim_avg.append(ssim_pred)
        tb.add_scalar('NATURE_MAE',mae_pred,idx)
        tb.add_scalar('NATURE_MSE',mse_pred,idx)
        tb.add_scalar('NATURE_RMSE',rmse_pred,idx)
        tb.add_scalar('NATURE_PSNR',psnr_pred,idx)
        tb.add_scalar('NATURE_SSIM',ssim_pred,idx)
        tb.add_text('NATURE_MAE',total_image_index[idx],idx)
        tb.add_text('NATURE_MSE',total_image_index[idx],idx)
        tb.add_text('NATURE_RMSE',total_image_index[idx],idx)
        tb.add_text('NATURE_PSNR',total_image_index[idx],idx)
        tb.add_text('NATURE_SSIM',total_image_index[idx],idx)
    mae_avg = np.mean(mae_avg)
    mse_avg = np.mean(mse_avg)
    rmse_avg = np.mean(rmse_avg)
    psnr_avg = np.mean(psnr_avg)
    ssim_avg = np.mean(ssim_avg)
    print('mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
          .format(mae_pred=mae_avg, mse_pred=mse_avg, rmse_pred=rmse_avg, psnr_pred=psnr_avg, ssim_pred=ssim_avg))

    #write metrics file
    save_txt_path = os.path.join(args.experiment_path ,args.exp ,args.date,args.model_name,'Nature_metrics.txt')
    with open(save_txt_path, 'a') as f:
        log_metrics = 'mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'.format(mae_pred=mae_avg, mse_pred=mse_avg, rmse_pred=rmse_avg, psnr_pred=psnr_avg, ssim_pred=ssim_avg)
        f.write(log_metrics)



