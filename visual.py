import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torchvision.transforms as T
import cv2
import visformer
from data.dataset import DatasetWithTextLabel
from PIL import Image
import matplotlib.pyplot as plt

def main(args):
    img_list = os.listdir("image")

    # prepare training and testing dataloader
    normalize = T.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    transforms = T.Compose([
                T.ToTensor(),
                T.Resize(256),
                T.CenterCrop(224),
                normalize,
            ])

    train_dataset = DatasetWithTextLabel(args.dataset, transforms, split='train')
    num_classes = len(train_dataset.dataset.classes)
    print("num_classes  :{}".format(num_classes))
    args.num_classes = num_classes


    student = visformer.visformer_tiny(num_classes=num_classes)
    student = student.cuda(args.gpu)
    if args.resume:
        checkpoint = torch.load(args.resume)
        student.load_state_dict(checkpoint['state_dict'])
    student.eval()


    with torch.no_grad():
        for _ in img_list:
            image = cv2.imread(os.path.join("image",_))
            #image = Image.open(os.path.join("image",_))
            input_tensor = transforms(image).unsqueeze(0).cuda(args.gpu)
            a = student(input_tensor)
            # from thop import profile
            # from thop import clever_format
            # input = input_tensor

            # flops, params = profile(student, inputs=(input, )) # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
            # flops, params = clever_format([flops, params], "%.3f") 
            # print(flops, params)
            visulize_attention_map_batch(image, student._attention_map(),_)

def visulize_attention_map(image, attention_mask):
    """
    img_batch = [bs,C,H,W] torch.tensor
    attention_mask: 2-D 的numpy矩阵
    """

    _,C,H,W = attention_mask.shape
    H = int(H**(0.5))
    print(f"mask shape:{attention_mask.shape}")
    attention_mask = attention_mask[0,0,1:22,:]
    attention_mask = torch.mean(attention_mask,dim=0).reshape(H,H)
    #attention_mask_0 = attention_mask[0,0,23,:].reshape(H,H)
    print(attention_mask.shape)
    attention_mask = attention_mask.cpu().detach().numpy()
    mask = Image.fromarray(attention_mask).resize((image.size))
    mask = mask / np.max(mask)
    mask = np.uint8(255 * mask)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(image)
    ax[0].axis('off')
    
    ax[1].imshow(image)#带红框和cls的原图
    ax[1].imshow(mask, alpha=0.2, cmap='rainbow')#注意力图
    ax[1].axis('off')
    plt.show()

def visulize_attention_map_batch(img, attention_mask,name):
    """
    img_batch = [bs,C,H,W] torch.tensor
    attention_mask: 2-D 的numpy矩阵
    """
    b,C,H,W = attention_mask.shape
    print(f"attention_mask:{attention_mask.shape}")
    H = int(H**(0.5))
    attention_mask = attention_mask[0,0,:,:]
    print(attention_mask.shape)
    attention_mask = torch.mean(attention_mask,dim=0).reshape(H,H)
    #attention_mask = attention_mask[0,0,18,:].reshape(H,H)

    attention_mask = attention_mask.cpu().detach().numpy()

    img_h, img_w = img.shape[0], img.shape[1]
    print(img.shape)
    mask = attention_mask
    print(mask.shape)
    mask = cv2.resize(mask, (img_w, img_h))
    normed_mask = mask / np.max(mask)
    normed_mask = np.uint8(255 * normed_mask)
    print(normed_mask.shape, img.shape)
    normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    normed_mask = cv2.addWeighted(img, 1, normed_mask, 0.6, 0)
    cv2.imshow(name,normed_mask)
    cv2.waitKey(1000)
    cv2.imwrite(f'visualization/tsne/zz{name}',normed_mask)
    cv2.imwrite(f'visualization/tsne/xx{name}',img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='QFSD', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'QFSD'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--resume', type=str, default='/home/iim321/lmy/my_sp/checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth')

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)