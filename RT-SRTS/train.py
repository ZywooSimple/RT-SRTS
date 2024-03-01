import argparse
from trainer import Trainer_ReconNet
from dataset_multiangle.offline_multiDataset import offline_multiDataSet
from dataset_multiangle.OurFn import our_gan
import tensorboardX as tbX
import torch 
import time
import os
import shutil

def parse_args():
  parse = argparse.ArgumentParser(description='PatRecon')
  parse.add_argument('--exp', type=str, default='zm_p2', dest='exp',
                     help='patient case ') 
  parse.add_argument('--date', type=str, default='xray_try_zm_seed_128d',help='the model name to save')
  parse.add_argument('--batch_size', type=int, default=1, dest='batch_size',
                     help='batch_size')
  parse.add_argument('--epoch', type=int, default=100, dest='epoch',
                     help='epoch')
  parse.add_argument('--lr', type=float, default=0.0002, dest='lr',
                     help='lr') 
  parse.add_argument('--gpu', type=str, default='0', dest='gpuid',
                     help='gpu is split by')    


  parse.add_argument('--experiment_path', type=str, default='/hdd2/zmx/experiment/')    
  parse.add_argument('--phase',type=str,default='train')                    
  parse.add_argument('--arch', type=str, default='ReconNet', dest='arch',
                     help='architecture of network')
  parse.add_argument('--print_freq', type=int, default=1, dest='print_freq',
                     help='print freq')
  parse.add_argument('--resume', type=str, default='final', dest='resume',
                     help='resume model')
  parse.add_argument('--num_views', type=int, default=1, dest='num_views',
                     help='none')
  parse.add_argument('--output_channel', type=int, default=128, dest='output_channel',
                     help='output_channel')
  parse.add_argument('--classes', type=int, default=2, dest='classes',
                     help='output_classes')
  parse.add_argument('--loss', type=str, default='l2', dest='loss',
                     help='loss')
  parse.add_argument('--optim', type=str, default='adam', dest='optim',
                     help='optim')
  parse.add_argument('--weight_decay', type=float, default=0, dest='weight_decay',
                     help='weight_decay')
  parse.add_argument('--init_gain', type=float, default=0.02, dest='init_gain',
                     help='init_gain')
  parse.add_argument('--init_type', type=str, default='standard', dest='init_type',
                     help='init_type')
  parse.add_argument('--save_epoch_freq', type=int, default=1, dest='save_epoch_freq',
                     help='save_epoch_freq')
  parse.add_argument('--model_name', type=str, default='best_model', dest='model_name',
                      help='model_name')
  parse.add_argument('--contrain', type=str, default=None, dest='contrain',
                      help='contrain')
  args = parse.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  # check gpu
  assert (torch.cuda.is_available())
  split_gpu = str(args.gpuid).split(',')
  args.gpu_ids = [int(i) for i in split_gpu]
  args.fine_size=128
  args.output_path = os.path.join(args.experiment_path ,args.exp ,args.date,'model')
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
  log_dir=os.path.join(args.experiment_path ,args.exp ,args.date,'loss')
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  args.dataroot=os.path.join('/hdd2/zmx',args.exp,'h5py')
  print(args.dataroot)
  train_datasetfile = os.path.join(args.experiment_path,'train.txt')
  valid_datasetfile =os.path.join(args.experiment_path,'val.txt')
  train_dataset = offline_multiDataSet(args,train_datasetfile)
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    collate_fn=our_gan)
  print('total training images: %d' % (len(train_dataloader)*args.batch_size))

  # valid dataset
  valid_dataset = offline_multiDataSet(args, valid_datasetfile)
  valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    collate_fn=our_gan)
  print('total validation images: %d' % len(valid_dataloader))


  #get model
  model=Trainer_ReconNet(args)
  start_epoch = 1
  if args.contrain is not None:
    start_epoch = model.load()+1

  tb = tbX.SummaryWriter(log_dir=log_dir)
  for epoch in range(start_epoch, args.epoch):
    #train
    start_time = time.time()
    train_recon_loss,train_seg_loss,train_loss=model.train_epoch(train_loader=train_dataloader,epoch=epoch,tb=tb)
    train_end_time = time.time()
    tb.add_scalar('Train_Recon_Loss', train_recon_loss, epoch)
    tb.add_scalar('Train_Seg_Loss', train_seg_loss, epoch)
    tb.add_scalar('Train_Loss', train_loss, epoch)
    

    #val
    val_recon_loss,val_seg_loss,val_loss=model.validate(val_loader=valid_dataloader)
    tb.add_scalar('Val_Recon_Loss', val_recon_loss, epoch)
    tb.add_scalar('Val_Seg_Loss', val_seg_loss, epoch)
    tb.add_scalar('Val_Loss', val_loss, epoch)

    # save curr model
    print('saving the model at the end of epoch %d' %epoch)
    model.save(val_loss, epoch=epoch)
    # save model several epoch
    if epoch%10==0:
      model.save_epoch(val_loss, epoch=epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec \t' %
          (epoch, args.epoch, time.time() - start_time))

