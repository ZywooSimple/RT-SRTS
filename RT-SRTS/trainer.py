import shutil
import os.path as osp
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import ReconNet
from utils import *
import functools
import math
class Trainer_ReconNet(nn.Module):
    def __init__(self, args):
        super(Trainer_ReconNet, self).__init__()

        self.exp_name = args.exp
        self.arch = args.arch
        self.print_freq = args.print_freq
        self.output_path = args.output_path
        self.resume = args.resume
        self.best_loss = 1e5
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        self.phase=args.phase
        # create model
        print("=> Creating Generator model...")
        if self.arch == 'ReconNet':
            self.model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel, gain=args.init_gain, init_type=args.init_type).to(self.device)
        else:
            assert False, print('Not implemented model: {}'.format(self.arch))

        # define loss function
        if args.loss == 'l1':
            # L1 loss
            self.criterion = nn.L1Loss(reduction='mean').to(self.device)
        elif args.loss == 'l2':
            # L2 loss (mean-square-error)
            self.criterion = nn.MSELoss(reduction='mean').to(self.device)
        else:
            assert False, print('Not implemented loss: {}'.format(args.loss))

        # segmentation loss
        self.criterionDice = DiceMeanLoss().to(self.device)
        self.bceloss=nn.BCELoss(reduction='mean').to(self.device)


        # define optimizer
        if args.optim == 'adam':
            self.optimizer_G = torch.optim.Adam(self.model.parameters(), 
                                            lr=args.lr,
                                            betas=(0.5, 0.999),
                                            weight_decay=args.weight_decay,  
                                            )
            
        else:
            assert False, print('Not implemented optimizer_G: {}'.format(args.optim))

        #define scheduler
        lambda_G=lambda epoch: 1 if epoch<=(args.epoch/2.0) else 1-(epoch-(args.epoch/2.0))/(args.epoch-(args.epoch/2.0))#生成器学习率调整机制
        self.scheduler_G=torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,lr_lambda=lambda_G)
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_epoch(self, train_loader, epoch, tb):
        train_recon_loss = AverageMeter()
        train_seg_loss=AverageMeter()
        train_loss = AverageMeter()
        train_start_time=time.time()
        # train mode
        self.model.train()
        #drr, ct, lung, tumor, backg,  file_path
        for i, (input, target,label,image_path) in enumerate(train_loader):     ########,angle
            input_var=input.to(self.device)   #[batch,2,128,128]
            target_var = target.to(self.device) #groud truth CT [batch,128,128,128]
            label=label.to(self.device) #label [batch,3,128,128,128]
            # compute output
            output,seg_output= self.model(input_var) # network output CT:[batch,128,128,128]    seg_output:[batch,2,128,128,128]

            # compute loss
            loss = 0.0
            #重建损失
            Recon_loss = self.criterion(output, target_var)
            loss += 1 * Recon_loss
            #分割损失
            Seg_loss=self.bceloss(seg_output,label)
            loss+=1*Seg_loss


            self.optimizer_G.zero_grad()
            loss.backward()
            self.optimizer_G.step()

            train_recon_loss.update(Recon_loss.data.item(), input.size(0))
            train_seg_loss.update(Seg_loss.data.item(), input.size(0))
            train_loss.update(loss.data.item(), input.size(0))

            # visualization
            tb.add_image('CT',target_var.cpu()[0,100,:,:],i,dataformats='HW')
            tb.add_image('generated_CT',output.detach().cpu()[0,100,:,:],i,dataformats='HW')



            # display info
            if i % self.print_freq == 0:
                print('Epoch: [{0}] \t'
                        'Iter: [{1}/{2}]\t'
                        'Recon Loss: {ReconLoss.val:.5f} ({ReconLoss.avg:.5f})\t'
                        'Seg Loss: {SegLoss.val:.5f} ({SegLoss.avg:.5f})\t'
                        'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'
                        .format(
                        epoch, i, len(train_loader),
                        ReconLoss = train_recon_loss,
                        SegLoss=train_seg_loss,
                        loss=train_loss))
        current_learning_rate= self.optimizer_G.param_groups[0]['lr']
        self.scheduler_G.step(epoch)
        next_epoch_learning_rate= self.optimizer_G.param_groups[0]['lr']
        # finish current epoch 
        print(f'learning rate change:{current_learning_rate}-->{next_epoch_learning_rate}')

        print('Finish Epoch: [{}]\t'
              'Train Time Taken: {train_time:.4f}\t'
              'Average Recon Loss: {ReconLoss.avg:.5f}\t'
              'Average Seg Loss: {SegLoss.avg:.5f}\t'
              'Average Train Loss: {loss.avg:.5f}\t'.format(
               epoch, train_time=time.time() - train_start_time,
               ReconLoss = train_recon_loss,
               SegLoss=train_seg_loss,
               loss=train_loss))

        return train_recon_loss.avg,train_seg_loss.avg,train_loss.avg


    def validate(self, val_loader):
        val_recon_loss = AverageMeter() 
        val_seg_loss = AverageMeter()
        val_loss = AverageMeter()

        # evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i, (input, target, label,image_path) in enumerate(val_loader):  #####,angle
                input_var = input.to(self.device) #[batch,1,128,128]
                target_var = target.to(self.device)#[batch,128,128,128]
                label=label.to(self.device)

                # compute output
                output,seg_output = self.model(input_var)

                # compute loss
                loss = 0.0
                Recon_loss = self.criterion(output, target_var)
                loss += 1 * Recon_loss

                Seg_loss=self.bceloss(seg_output,label)
                loss+=1*Seg_loss

                val_recon_loss.update(Recon_loss.data.item(), input.size(0))
                val_seg_loss.update(Seg_loss.data.item(), input.size(0))
                val_loss.update(loss.data.item(), input.size(0)) #当前的验证损失只计算重建加分割的最佳网络
        return val_recon_loss.avg,val_seg_loss.avg,val_loss.avg


    def save(self, curr_val_loss, epoch):
        # update best loss and save checkpoint
        is_best = curr_val_loss < self.best_loss
        self.best_loss = min(curr_val_loss, self.best_loss)

        state = {'epoch': epoch + 1,
                'arch': self.arch,
                'state_dict': self.model.state_dict(),
                'best_loss': curr_val_loss
                }

        filename = osp.join(self.output_path, 'curr_model.pth.tar')
        best_filename = osp.join(self.output_path, 'best_model.pth.tar')

        print('! Saving checkpoint: {}'.format(filename))
        torch.save(state, filename)

        if is_best:
            print('!! Saving best checkpoint: {}'.format(best_filename))
            shutil.copyfile(filename, best_filename)

    def save_epoch(self, curr_val_loss, epoch):
        # update best loss and save checkpoint

        state = {'epoch': epoch + 1,
                'arch': self.arch,
                'state_dict': self.model.state_dict(),
                'best_loss': curr_val_loss
                }

        filename = osp.join(self.output_path, '%d_model.pth.tar'%(epoch))

        print('! Saving checkpoint: {}'.format(filename))
        torch.save(state, filename)


    def load(self):

        if self.resume == 'best':
            ckpt_file = osp.join(self.output_path, 'best_model.pth.tar')
        elif self.resume == 'final':
            ckpt_file = osp.join(self.output_path, 'curr_model.pth.tar')
        elif self.resume is not None:
            ckpt_file = osp.join(self.output_path, self.resume+'_model.pth.tar')
        else:
            assert False, print("=> no available checkpoint")

        if osp.isfile(ckpt_file):
            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file)
            start_epoch = checkpoint['epoch']

            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))

        return start_epoch


