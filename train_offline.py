import argparse
import os
import numpy as np
import cv2.cv2 as cv2
from tqdm import tqdm
import blockline
import torch
import sys
# sys.path.append(r'/home/user1106/DeepNets/Deeplab-pytorch')
print(sys.path)

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.RailTrack import RTrackNet
from saveimage import saveimage


NUM_CLASSES = 4
class Trainer(object):

    def __init__(self, args):
        self.args = args

        #1----- Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        self.args.outputFile = os.path.join(self.saver.experiment_dir,'output' )
        if not os.path.exists(self.args.outputFile):
            os.makedirs(self.args.outputFile)    #set output path

            
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        #2----- Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        print('class number: ',self.nclass)

        #3----- Define network      
        model = RTrackNet(backbone=args.backbone,
                            neck=args.neck, 
                            output_stride=16, 
                            inChannal= 4,
                            num_classes= NUM_CLASSES)

        #4-----Define train params
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        #5----- Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        #6----- Define Loss function
            #6.1----- whether to use class balanced weights
        if args.use_balanced_weights:
            weight = torch.from_numpy(np.array([1.75, 9,7.6, 15]).astype(np.float32))
        else:
            weight = None
            #6.2------ Loss function
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

            #6.3------ DFR function
        self.DFR = blockline.railSegmenter()
        
        self.model, self.optimizer = model, optimizer        
        
        #7----- Define Evaluator
        self.evaluator_MIOU = Evaluator(self.nclass)
        self.evaluator_RIOU = Evaluator(2)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        #8----- Using cuda
        if args.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
            print("GPU CNT: ",torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        #9------  Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            #9.1--- load checkpoint
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            #9.2--- start epoch
            args.start_epoch = checkpoint['epoch']
            #9.3--- load model
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            #9.4--- loding optimizer
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        #10-----  Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()  
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            #1-----sample
            image, target, mask = sample['image'], sample['label'],sample['mask']
            
            if self.args.cuda:
                image, target,mask = image.cuda(), target.cuda(), mask.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            #2------train & update
            self.optimizer.zero_grad()

            input = torch.cat((image,mask.unsqueeze(1)), dim= 1)
            output = self.model(input)
            loss = self.criterion(output.contiguous(), target.contiguous()) 
            loss.backward()                         
            self.optimizer.step()                   #update model

            #3-------record
            train_loss += loss.item() #record loss
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        #summary for one epoch
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator_MIOU.reset()
        self.evaluator_RIOU.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            
            image, target, mask = sample['image'], sample['label'],sample['mask']
            
            if self.args.cuda:
                image, target,mask = image.cuda(), target.cuda(), mask.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            input = torch.cat((image,mask.unsqueeze(1)), dim= 1)

            with torch.no_grad():
                output = self.model(input)

            loss = self.criterion(output, target) 
            test_loss += loss.item()
            tbar.set_description('Validate loss: %.3f' % (test_loss / (i + 1)))

            pred = output.data.cpu().numpy()  # pred
            target = target.cpu().numpy()       # ground truth
            pred = np.argmax(pred, axis=1)      # pred mask
            self.evaluator_MIOU.add_batch(target, pred)
            
            _target = target
            _target[_target<2] = 0
            _target[_target>=2] = 1
            _pred = pred
            _pred = _pred.astype('uint8')
            _pred[_pred<2] = 0
            _pred[_pred>=2] = 1
            # _pred = self.DFR.run4train(_pred)  #******DFR
            self.evaluator_RIOU.add_batch(_target, _pred.astype('uint8'))
        
        Acc = self.evaluator_MIOU.Pixel_Accuracy()
        Acc_class = self.evaluator_MIOU.Pixel_Accuracy_Class()
        mIoU = self.evaluator_MIOU.Mean_Intersection_over_Union()
        FWIoU = self.evaluator_MIOU.Frequency_Weighted_Intersection_over_Union()

        Acc2 = self.evaluator_RIOU.Pixel_Accuracy()
        Acc_class2 = self.evaluator_RIOU.Pixel_Accuracy_Class()
        mIoU2 = self.evaluator_RIOU.Mean_Intersection_over_Union()
        FWIoU2 = self.evaluator_RIOU.Frequency_Weighted_Intersection_over_Union()

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)

        for iid in range(self.nclass):
            self.writer.add_scalar('val/fwIoU{}'.format(iid), FWIoU[iid], epoch)
            self.writer.add_scalar('val/mIoU{}'.format(iid), mIoU[iid], epoch)
            self.writer.add_scalar('val/Acc_class{}'.format(iid), Acc_class[iid], epoch)
        for iid in range(2):
            self.writer.add_scalar('val/fwIoU{}'.format(iid), FWIoU2[iid], epoch)
            self.writer.add_scalar('val/mIoU{}'.format(iid), mIoU2[iid], epoch)
            self.writer.add_scalar('val/Acc_class{}'.format(iid), Acc_class2[iid], epoch)
              

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc2, Acc_class2, mIoU2, FWIoU2))
        print('Loss: %.3f' % test_loss)
        
        with open(os.path.join(self.args.outputFile, 'epoch_result.txt'), 'a') as f:
            f.write('Validation:\n')
            f.write('[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            f.write("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}\n".format(Acc, Acc_class, mIoU, FWIoU))
            f.write("Acc2:{}, Acc_class2:{}, mIoU2:{}, fwIoU2: {}\n".format(Acc2, Acc_class2, mIoU2, FWIoU2))
            
            f.write('Loss: %.3f\n' % test_loss)
            f.write('IOU: %.3f\n' % mIoU.mean())

        new_pred = mIoU2[1].mean()

        if new_pred > self.best_pred:
            print("****save model***")
            torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'best_pred':self.best_pred,
                         }, 
                    os.path.join(self.saver.experiment_dir,'best_model.pth'))

            self.best_pred = new_pred
            with open(os.path.join(self.args.outputFile, 'best_pred.txt'), 'w') as f:
                f.write(str(new_pred))

    #=========测试
    def test(self, epoch):
        print(self.args.outputFile)

        self.model.eval()
        if epoch > 0:
            tbar = tqdm(self.val_loader, desc='\r')
        else:
            tbar = tqdm(self.test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            
            image, mask = sample['image'],sample['mask']
            
            if self.args.cuda:
                image,mask = image.cuda(), mask.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            input = torch.cat((image,mask.unsqueeze(1)), dim= 1)

            with torch.no_grad():
                output = self.model(input)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            
            saveimage(image.cpu().permute(0,2,3,1).numpy() ,
                                    pred,
                                    root=self.args.outputFile,
                                    subroot='ep%03d'%epoch,
                                    startID=i*self.args.batch_size, 
                                    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='CSPnet',
                        # choices=['resnet', 'xception', 'drn', 'mobilenet','CSPnet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--neck', type=str, default='A-S',
                        help='neck name (default: A-S)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='iRailway',
                        help='dataset name (default: iRailway)')
    parser.add_argument('--dataAug', type=str, default='no',
                        choices=['no','all'],
                        help='data argument type(default: all)')

    parser.add_argument('--dataRepeat', type=int, default=1)

    parser.add_argument('--trainRepeat', type=int, default=1)

    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=True,
                        help='skip validation during training')

    parser.add_argument('--add-opp', action='store_true', default=True,
                        help='add opp sample')


    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    

    if args.sync_bn is None:
        if args.cuda and len(gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.epochs is None:
        args.epochs = 30

    if args.batch_size is None:
        args.batch_size = 4 * len(gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        args.lr = 0.01 / (4 * len(gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = args.backbone+'-'+args.neck+'-'+args.loss_type+'-'+args.dataset
    

    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if  epoch % 5 == 4:
            trainer.validation(epoch) 
        if  epoch % 5 == 0:     
            trainer.test(epoch)
    trainer.test(0)

    trainer.writer.close()

if __name__ == "__main__":

    
    main()

