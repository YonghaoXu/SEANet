import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import os.path as osp
import time
from utils.tools import *
from dataset.synthia_dataset import synthiaDataSet
from dataset.cityscapes16_dataset import cityscapes16DataSet
from model.SEAN import SEANet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)



def get_arguments():

    parser = argparse.ArgumentParser(description="SEAN")
    
    #dataset
    parser.add_argument("--data_dir_source", type=str, default='/data/yonghao.xu/SegmentationData/synthia/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/synthia_imagelist_train_.txt',
                        help="source dataset list file.")
    parser.add_argument("--data_dir_target", type=str, default='/data/yonghao.xu/SegmentationData/cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list_target", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size", type=str, default='1024,512',
                        help="width and height of input images.")    
    parser.add_argument("--input_size_target", type=str, default='2048,1024',
                        help="width and height of target images.")                        
    parser.add_argument("--num_classes", type=int, default=16,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--restore_from", type=str, default='/data/yonghao.xu/PreTrainedModel/fcn8s_from_caffe.pth',
                        help="pretrained VGG model.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="noise.")

    #hyperparameters
    parser.add_argument("--teacher_alpha", type=float, default=0.99,
                        help="teacher alpha in EMA.")
    parser.add_argument("--attention_threshold", type=float, default=0.3,
                        help="attention threshold.")
    parser.add_argument("--st_weight", type=float, default=0.3,
                        help="self-ensembling weight.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./Snap/',
                        help="where to save snapshots of the model.")

    
    return parser.parse_args()


def main():

    """Create the model and start the training."""
    args = get_arguments()
    if os.path.exists(args.snapshot_dir)==False:
        os.mkdir(args.snapshot_dir)
    f = open(args.snapshot_dir+'Synthia2Cityscapes_log.txt', 'w')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    # Create network
    student_net = SEANet(num_classes=args.num_classes)
    teacher_net = SEANet(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)

    new_params = student_net.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
            new_params[j] = saved_state_dict[i]

    student_net.load_state_dict(new_params)
    teacher_net.load_state_dict(new_params)
    
    
    for name, param in teacher_net.named_parameters():
        param.requires_grad=False

    teacher_net = teacher_net.cuda()
    student_net = student_net.cuda()

   

    src_loader = data.DataLoader(
                    synthiaDataSet(args.data_dir_source, args.data_list_source,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    tgt_loader = data.DataLoader(
                    cityscapes16DataSet(args.data_dir_target, args.data_list_target, max_iters=9400,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    val_loader = data.DataLoader(
                    cityscapes16DataSet(args.data_dir_target, args.data_list_target, max_iters=None,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)


    num_batches = min(len(src_loader),len(tgt_loader))
    
    optimizer = optim.Adam(student_net.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())

    teacher_optimizer = WeightEMA(
        teacher_params, 
        student_params,
        alpha=args.teacher_alpha,
    )



    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    n_class = args.num_classes
    num_steps = args.num_epoch*num_batches
    loss_hist = np.zeros((num_steps,5))
    index_i = -1
    OA_hist = 0.2
    aug_loss = torch.nn.MSELoss()

    for epoch in range(args.num_epoch):
        if epoch==6:
            return
        for batch_index, (src_data, tgt_data) in enumerate(zip(src_loader, tgt_loader)):
            index_i += 1
       
            tem_time = time.time()
            student_net.train()
            optimizer.zero_grad()

            # train with source
            images, src_label, _, im_name = src_data
            images = images.cuda()
            src_label = src_label.cuda()            
            _,src_output = student_net(images)
            src_output = interp(src_output)
            # Segmentation Loss
            cls_loss_value = loss_calc(src_output, src_label)
            _, predict_labels = torch.max(src_output, 1)
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = src_label.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch.append(mean_iu)                
            miu = np.mean(metrics_batch, axis=0)  

            
            # train with target
            images, label_target,_, im_name = tgt_data
            images = images.cuda()
            label_target = label_target.cuda()
            tgt_t_input = images + torch.randn(images.size()).cuda() * args.noise
            tgt_s_input = images + torch.randn(images.size()).cuda() * args.noise

            _,tgt_s_output = student_net(tgt_s_input)
            t_confidence,tgt_t_output = teacher_net(tgt_t_input)

            t_confidence = t_confidence.squeeze()
           
            # self-ensembling Loss
            tgt_t_predicts = F.softmax(tgt_t_output, dim=1).transpose(1, 2).transpose(2, 3)
            tgt_s_predicts = F.softmax(tgt_s_output, dim=1).transpose(1, 2).transpose(2, 3)

            
            mask = t_confidence > args.attention_threshold
            mask = mask.view(-1)
            num_pixel = mask.shape[0]
            
            mask_rate = torch.sum(mask).float() / num_pixel

            tgt_s_predicts = tgt_s_predicts.contiguous().view(-1,n_class)
            tgt_s_predicts = tgt_s_predicts[mask]
            tgt_t_predicts = tgt_t_predicts.contiguous().view(-1,n_class)
            tgt_t_predicts = tgt_t_predicts[mask]
            aug_loss_value = aug_loss(tgt_s_predicts, tgt_t_predicts)
            aug_loss_value = args.st_weight * aug_loss_value

            # TOTAL LOSS
            if mask_rate==0.:
                aug_loss_value = torch.tensor(0.).cuda()
            
            total_loss = cls_loss_value + aug_loss_value
            
            total_loss.backward()
            loss_hist[index_i,0] = total_loss.item()
            loss_hist[index_i,1] = cls_loss_value.item()
            loss_hist[index_i,2] = aug_loss_value.item()
            loss_hist[index_i,3] = miu
                    
            optimizer.step()
            teacher_optimizer.step()
            batch_time = time.time()-tem_time

            if (batch_index+1) % 10 == 0: 
                print('epoch %d/%d:  %d/%d time: %.2f miu = %.1f cls_loss = %.3f st_loss = %.3f \n'%(epoch+1, args.num_epoch,batch_index+1,num_batches,batch_time,np.mean(loss_hist[index_i-9:index_i+1,3])*100,np.mean(loss_hist[index_i-9:index_i+1,1]),np.mean(loss_hist[index_i-9:index_i+1,2])))
                f.write('epoch %d/%d:  %d/%d time: %.2f miu = %.1f cls_loss = %.3f st_loss = %.3f \n'%(epoch+1, args.num_epoch,batch_index+1,num_batches,batch_time,np.mean(loss_hist[index_i-9:index_i+1,3])*100,np.mean(loss_hist[index_i-9:index_i+1,1]),np.mean(loss_hist[index_i-9:index_i+1,2])))
                f.flush() 
                
            if (batch_index+1) % 500 == 0:                 
                OA_new = test_mIoU16(f,teacher_net, val_loader, epoch+1,input_size_target,print_per_batches=10)
                
                # Saving the models        
                if OA_new > OA_hist:    
                    f.write('Save Model\n') 
                    print('Save Model')                     
                    model_name = 'Synthia2Cityscapes_epoch'+repr(epoch+1)+'batch'+repr(batch_index+1)+'tgt_miu_'+repr(int(OA_new*1000))+'.pth'
                    torch.save(teacher_net.state_dict(), os.path.join(
                        args.snapshot_dir, model_name))   
                    OA_hist = OA_new


    f.close()
    torch.save(teacher_net.state_dict(), os.path.join(
        args.snapshot_dir, 'Synthia_TeacherNet.pth'))
    np.savez(args.snapshot_dir+'Synthia_loss.npz',loss_hist=loss_hist) 
    


if __name__ == '__main__':
    main()
