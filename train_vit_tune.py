import os
import time
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import clip
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import visformer
from data.dataloader import EpisodeSampler, MultiTrans
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval



def main(args):
    # checkpoint and tensorboard dir
    args.tensorboard_dir = 'tensorboard/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    args.checkpoint_dir = 'checkpoint/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])
    if args.aug_support > 1:
        aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                  # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  norm])
        test_aug = MultiTrans([test_aug] + [aug]*(args.aug_support-1))

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)
    print(num_classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    if args.nlp_model == 'clip':
        teacher, _ = clip.load("ViT-B-32.pt", device='cuda:' + str(args.gpu))
        text_dim = 512
        # set the max text length
        if args.text_length != -1:
            teacher.context_length = args.text_length
            teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
            for layer in teacher.transformer.resblocks:
                layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]

    train_text = get_text_feature(teacher, train_dataset, args)
    test_text = get_text_feature(teacher, test_dataset, args)
    if args.eqnorm:
        if args.nlp_model in ['mpnet', 'glove']:
            # the bert features have been normalized to unit length. use the avg norm of clip text features
            avg_length = 9.
        else:
            avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        train_text = F.normalize(train_text, dim=-1) * avg_length
        test_text = F.normalize(test_text, dim=-1) * avg_length

    if args.model == 'visformer-t':
        model = visformer.visformer_tiny(num_classes=num_classes,mode = "tune")
        model_b = visformer.visformer_tiny(num_classes=num_classes,mode = "tune")
    else:
        raise ValueError(f'unknown model: {args.model}')

    tuning_model = visformer.Token_Attention_2(args.head,384,512,args=args)
    tuning_model_b = visformer.Token_Attention_2(args.head,384,512,args=args)
    model = model.cuda(args.gpu)
    tuning_model = tuning_model.cuda(args.gpu)
    model_b = model_b.cuda(args.gpu)
    tuning_model_b = tuning_model_b.cuda(args.gpu)
    # for name, param in model.named_parameters():
    #     param.requires_grad = False

    if args.optim == 'sgd':
        optim = torch.optim.SGD(tuning_model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.meta_main_lr, 'weight_decay': args.weight_decay},
                                   {'params': tuning_model.parameters(), 'lr': args.meta_tune_lr}], weight_decay=5e-7)
        optim_task = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.task_main_lr, 'weight_decay': args.weight_decay},
                                   {'params': tuning_model.parameters(), 'lr': args.task_tune_lr}], weight_decay=5e-6)
    else:
        raise ValueError(f'unknown optim: {args.optim}')




    if args.resume:
        args.init = args.resume
    if args.init:
        checkpoint = torch.load(args.init, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        raise ValueError('must provide pre-trained model')

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')

    if args.test:
        test(test_text, model, tuning_model, test_loader, 0, args)
        return

    best_acc = 0.
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        train(train_text, model, model_b,tuning_model, tuning_model_b,train_loader, optim,optim_task, epoch+1, args)

        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, model,model_b, tuning_model, tuning_model_b,test_loader, epoch+1, args)

        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optim.state_dict(),
        # }
        #torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_latest.pth')
        # if (epoch + 1) % args.save_freq == 0:
        #     torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}_{args.way}_{args.shot}.pth')
        # if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
        #     best_acc = acc
        #     torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_best.pth')
    end = time.time()
    t = end - start
    hour = t//3600
    min = (t - 3600*hour)//60
    print('train time in {}epochs:  {}hours {}min {}sec'.format(args.epochs,hour,min,min%60))
    


def get_text_feature(teacher, dataset, args):
    class_idx = dataset.dataset.classes
    idx2text = dataset.idx2text
    if args.no_template:
        text = [idx2text[idx] for idx in class_idx]
    else:
        text = ['A photo of ' + idx2text[idx] for idx in class_idx]
        #print(text)

    teacher.eval()
    if args.nlp_model == 'clip':
        text_token = clip.tokenize(text).cuda(args.gpu)
        if args.text_length != -1:
            text_token = text_token[:, :args.text_length]
        with torch.no_grad():
            text_feature = teacher.encode_text(text_token)
            text_feature = text_feature.float()
    return text_feature


def train(text, model,model_b, tuning_model, tuning_model_b,train_loader, optim,optim_task,epoch, args):

    # for param_tensor in model.state_dict():
    #     #打印 key value字典
    #     print(param_tensor,'\t',model.state_dict()[param_tensor].size())
    # for param_tensor in model.state_dict():
    #     model_b.state_dict()[param_tensor] = model.state_dict()[param_tensor]
    # model_b.load_state_dict(model.state_dict())
    # tuning_model_b.load_state_dict(tuning_model.state_dict()) 
    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    losses = 0.
    accs = 0.
    for idx, episode in enumerate(train_loader):
        model_b.load_state_dict(model.state_dict())
        tuning_model_b.load_state_dict(tuning_model.state_dict()) 

        model.train()
        tuning_model.train()

        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        glabels = episode[1].cuda(args.gpu) # way * (shot+15)
        text_index = glabels.view(args.train_way, -1)[:,0] # 5way

        support_Q = text[text_index].unsqueeze(0).repeat(args.train_way * args.shot, 1, 1)#25,5,512
        

#    with torch.no_grad():
        _, sup_img_token = model(sup)
        #sup_img_token[25,49,384]   que_img_token[75,49,384] 
        sup_features = tuning_model(sup_img_token, support_Q)#25,5,512
        parameters = sum(p.numel() for p in tuning_model.parameters() if p.requires_grad)
        print('number of parameters:{}'.format(parameters))
        sim = F.normalize(sup_features, dim=-1).permute(1,0,2) @ F.normalize(text[text_index], dim=-1).unsqueeze(0).permute(1,2,0)
        #sim = sup_features.permute(1,0,2) @ text[text_index].unsqueeze(0).permute(1,2,0)
        sim = sim.squeeze().t()
        support_labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, args.shot).view(-1).cuda(args.gpu)
        loss = F.cross_entropy(sim / args.t, support_labels)

        optim_task.zero_grad()
        loss.backward()
        optim_task.step()

        _, que_img_token = model(que)
        # from thop import profile
        # from thop import clever_format
        # input = que
        # flops, params = profile(model, inputs=(input, )) # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
        # flops, params = clever_format([flops, params], "%.3f") 
        # print(flops, params)

        query_Q = text[text_index].unsqueeze(0).repeat(args.train_way * 15, 1, 1)#75,5,512
        que_features = tuning_model(que_img_token, query_Q)#75,5,512
        # flops, params = profile(tuning_model, inputs=(que_img_token, query_Q)) # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
        # flops, params = clever_format([flops, params], "%.3f") 
        # print(flops, params)

        sim = F.normalize(que_features, dim=-1).permute(1,0,2) @ F.normalize(text[text_index], dim=-1).unsqueeze(0).permute(1,2,0)
        #sim = que_features.permute(1,0,2) @ text[text_index].unsqueeze(0).permute(1,2,0)
        sim = sim.squeeze().t()
        
        query_labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)
        loss = F.cross_entropy(sim / args.t, query_labels)
        losses += loss.item()
        _, pred = sim.max(-1)
        accs += query_labels.eq(pred).sum().float().item() / query_labels.shape[0]


        optim.zero_grad()
        loss.backward()
        model.load_state_dict(model_b.state_dict())
        tuning_model.load_state_dict(tuning_model_b.state_dict()) 
        optim.step()        

        if (idx+1) % args.print_step == 0 or (idx+1) == len(train_loader):
            print_string = f'Train epoch: {epoch}, step: {(idx+1):3d}, loss: {losses / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}'
            print(print_string)


def test(text, model, model_b,tuning_model, tuning_model_b,test_loader,epoch, args):
    start = time.time()
    accs = []

    model_b.load_state_dict(model.state_dict())
    tuning_model_b.load_state_dict(tuning_model.state_dict())  
    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.task_main_lr, 'weight_decay': args.weight_decay},
                                   {'params': tuning_model.parameters(), 'lr': args.task_tune_lr}], weight_decay=5e-6)
    for idx, episode in enumerate(test_loader):
        start = time.time()
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        glabels = episode[1].cuda(args.gpu) # way * (shot+15)
        text_index = glabels.view(args.train_way, -1)[:,0] # 5way
        support_Q = text[text_index].unsqueeze(0).repeat(args.train_way * args.shot, 1, 1)#25,5,512
        query_Q = text[text_index].unsqueeze(0).repeat(args.train_way * 15, 1, 1)#75,5,512
        query_labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)
        best_acc = 0
        for i in range(args.tune_times):
            model.train()
            tuning_model.train()
            _, sup_img_token = model(sup)
            #sup_img_token[25,49,384]   que_img_token[75,49,384] 
            sup_features = tuning_model(sup_img_token, support_Q)#25,5,512
            sim = F.normalize(sup_features, dim=-1).permute(1,0,2) @ F.normalize(text[text_index], dim=-1).unsqueeze(0).permute(1,2,0)
            #sim = sup_features.permute(1,0,2) @ text[text_index].unsqueeze(0).permute(1,2,0)
            sim = sim.squeeze().t()
            support_labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, args.shot).view(-1).cuda(args.gpu)
            loss = F.cross_entropy(sim / args.t, support_labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            model.eval()
            tuning_model.eval()

            with torch.no_grad():
                _, que_img_token = model(que)
        
                
                que_features = tuning_model(que_img_token, query_Q)#75,5,512
                sim = F.normalize(que_features, dim=-1).permute(1,0,2) @ F.normalize(text[text_index], dim=-1).unsqueeze(0).permute(1,2,0)
                sim = sim.squeeze().t()
                _, pred = sim.max(-1)
                acc = query_labels.eq(pred).sum().float().item() / query_labels.shape[0]
                if acc > best_acc:
                    best_acc = acc
        end = time.time()
        print(end - start)         
        accs.append(best_acc)

        model.load_state_dict(model_b.state_dict())
        tuning_model.load_state_dict(tuning_model_b.state_dict()) 

    end = time.time()
    t = int(end - start)
    print('test time :  {}min {}sec'.format(t//60,t%60))
    m, h = mean_confidence_interval(accs)
    print(f'Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    
    args.logger.add_scalar('test/acc', m * 100, epoch)

    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='sp_5shot')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='QFSD', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'QFSD'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--aug_support', type=int, default=1)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    parser.add_argument('--meta_main_lr', type=float, default=1e-4)
    parser.add_argument('--meta_tune_lr', type=float, default=3e-4)
    parser.add_argument('--task_main_lr', type=float, default=1e-4)
    parser.add_argument('--task_tune_lr', type=float, default=3e-4)
    parser.add_argument('--head', type=int, default=8)
    
    parser.add_argument('--init', type=str, default='checkpoint/QFSD/visformer-t/pre-train/checkpoint_epoch_300.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--tune_times', type=int, default=3)
    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)

