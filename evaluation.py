import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.SEAN import SEANet
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tools import *

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SEAN")
    parser.add_argument("--data_dir", type=str, default='/data/yonghao.xu/SegmentationData/cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/data/yonghao.xu/PreTrainedModel/GTA2Cityscapes.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='./Snap/Maps',
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir+'Evaluation.txt', 'w')
    
    model = SEANet(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    input_size_target = (2048,1024)
    interp = nn.Upsample(size=(1024,2048), mode='bilinear')

    test_mIoU(f,model, testloader, 0,input_size_target,print_per_batches=10)
    
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _,_, name = batch
        _,output = model(image.cuda())
        output = interp(output).cpu().data[0].numpy()
    

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (args.snapshot_dir, name))
        output_col.save('%s/%s_color.png' % (args.snapshot_dir, name.split('.')[0]))
    
    f.close()

if __name__ == '__main__':
    main()
