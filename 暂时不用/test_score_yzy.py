#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import time
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt 
import os
import sys
import warnings
import logging
from thop import profile
logging.getLogger("profile").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
from models.Swin_Transformer_v2 import swinv2_tiny_window8_256
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batchSize', type=int, default=1, metavar='N',      #改一次送的图片张数
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N')    #整个数据集迭代多少次
# parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')    #学习速率,往小改
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')    #正则化
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print(args)
def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
set_random_seed(args.seed)          
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")
print("===> load datasets")


print ("===> load datasets")
data_transform = transforms.Compose([
    transforms.Resize((256,256)), 
    transforms.ToTensor()
])
    
print("====>load testdatset ")
test_data_root='data/val/'
test_dataset = ImageFolder(test_data_root,transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=1,
                                         shuffle=True, num_workers=args.threads)
classes_num=len(test_dataset .class_to_idx)
img_name=test_dataset .imgs
classes_name=test_dataset .class_to_idx
print('!!!!',classes_name)
# {'Bacterialblight': 0, 'Blast': 1, 'Brownspot': 2, 'Tungro': 3}


def test():
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
        test_acc2 = 0.0
        test_acc3 = 0.0

       
        total =0
        total0 =0
        total1 =0
        total2 =0
        total3 =0

        tmp_prediction=[]
        tmp_scores=[]
        total_label=[]
        total_features=[]
        for iteration, (images, labels) in enumerate(test_loader):
            # if iteration == 500:
            #     break
            images=images.to(device)
           
            labels=labels.to(device)
    
            # outputs, feature= model(images)
            outputs = model(images)
            outputs=F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
            
            index0=(labels == 0).nonzero()
            total0 += index0.size(0)
            test_acc0 += torch.sum(prediction[index0] == labels[index0])
            index1=(labels == 1).nonzero()
            total1 += index1.size(0)
            test_acc1 += torch.sum(prediction[index1] == labels[index1])
            index2=(labels == 2).nonzero()
            total2 += index2.size(0)
            test_acc2 += torch.sum(prediction[index2] == labels[index2])
            index3=(labels == 3).nonzero()
            total3+= index3.size(0)
            test_acc3 += torch.sum(prediction[index3] == labels[index3])
            
            print("===> Epoch[{}] =====>Mean_val_Acc:{:.4f},0_Acc:{:.4f},1_Acc:{:.4f},2_Acc:{:.4f},3_Acc:{:.4f}".format(
                iteration, test_acc/total,test_acc0/total0,test_acc1/total1,test_acc2/total2,test_acc3/total3))
            print(" =====pred 0:{}".format(np.array(prediction[index0].cpu().detach().numpy()).T))
            print(" =====pred 1:{}".format(np.array(prediction[index1].cpu().detach().numpy()).T))
            print(" =====pred 2:{}".format(np.array(prediction[index2].cpu().detach().numpy()).T))
            print(" =====pred 3:{}".format(np.array(prediction[index3].cpu().detach().numpy()).T))
           
            prediction= prediction.cpu().data.numpy()
            tmp_prediction.append(prediction)
            tmp_scores.append(outputs[0].data.cpu().numpy())
            total_label.append(labels[0].data.cpu().numpy())
            # feature = feature.view(images.size(0), -1)
            # feature=feature.data.cpu().numpy()
            # total_features.extend(feature)
            
           
    # return test_acc/total,np.array(tmp_prediction),np.array(total_label),np.array(tmp_scores),np.array(total_features)
    return test_acc/total,np.array(tmp_prediction),np.array(total_label),np.array(tmp_scores)

        
if __name__ == '__main__' :
       import os
       from sklearn.metrics import confusion_matrix
       import pandas as pd
       from sklearn.preprocessing import label_binarize
       from sklearn.metrics import precision_recall_curve
       from sklearn.metrics import roc_curve, auc
       from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support
       from scipy import interp
       import matplotlib as mpl
       import matplotlib.pyplot as plt 

       print('===> Building model')

       model = swinv2_tiny_window8_256()

       num_ftrs = model.cls_head.cls.in_features
       model.cls_head.cls = nn.Linear(num_ftrs, 4)
        
       model = model.to(device)
       # model.load_state_dict(torch.load('checkpoints/SwinV2/best.pth',map_location={'cuda:1':'cuda:0'}))
       model = torch.load('checkpoints/SwinV2/best.pth')
       model.eval()

       model_names=['our']
       for n in range(len(model_names)):

            save_path='results/{}/'.format(model_names[n])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model = model.to(device)
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.2fM" % (total/1e6))
 
#            from torchstat import stat
#            inputs = torch.randn(1, 1, 256, 256)
#            flops, params = profile(model, inputs=(inputs.to(device), ))
#            print("{}:,Flops:{}G ,Params1:{} M ".format(model_names[n],flops/1e9, params/1e6))
          
            #stat(model, (1, 128, 128))
            if 1:
                log_dir='checkpoints/{}/best.pth'.format(model_names[n])
                last_log='checkpoints/{}/seq_last.pth'.format(model_names[n])
                best_log='checkpoints/{}/seq_best.pth'.format(model_names[n])
                # model.load_state_dict(torch.load(log_dir,map_location={'cuda:0':'cuda:1'}))
                model = torch.load('checkpoints/SwinV2/best.pth')
                print('load_weight')
                model.to(device)  
              
              	#!!!
                # test_acc,total_pred,total_label, total_scores,total_features=test()
                test_acc,total_pred,total_label, total_scores=test()

                #print('!!!!!!!!!!!!!!!!\ntest acc:{}'.format(test_acc))
                #print('!!!!!!!!!!!!!!!!\ntotal scores:{}'.format(total_scores()))
                if 1:
                    from openTSNE import TSNE
                    from sklearn.manifold import TSNE

                    '''
                    embedding = TSNE(n_components=2,init = 'pca',verbose = 1).fit_transform(total_features)
                    fig=plt.figure()
                    #mpl.colors.ListedColormap
                    cm = mpl.colors.ListedColormap(['m','r','b','g'])
                    #plt.scatter(embedding[:,0],embedding[:,1], c=total_label,s=1,cmap=plt.cm.get_cmap("rainbow", 10)) 
                    plt.scatter(embedding[:,0],embedding[:,1], c=total_label,s=1,cmap=plt.cm.gist_rainbow) 
                    #save_name='results/{}/tsne.tiff'.format(model_names[n])
                    save_name='tsne.tif'.format(model_names[n])
                    plt.savefig(save_name,dpi=300)
                    plt.show()

                   
                    save_csv_name='results/{}/tsne.xlsx'.format(model_names[n])
                    Cell_Features=[total_label,embedding[:,0],embedding[:,1]]
                    with pd.ExcelWriter(save_csv_name) as writer:
                            names=['Labels','Features1','Features2']
                            df =pd.DataFrame( data= np.array(Cell_Features).T ,columns=names)
                            df.to_excel(writer)
                    confusion_matrix_test = confusion_matrix(total_label, total_pred, labels=None, sample_weight=None)
                    confusion_matrix_test=confusion_matrix_test
                    confusion_matrix_acc= confusion_matrix_test.astype('float')/confusion_matrix_test.sum(axis=1)[:, np.newaxis]
                    save_csv_name='results/{}/confusion_matrix_acc_{}.xlsx'.format(model_names[n],str(test_acc))

                    df =pd.DataFrame(confusion_matrix_acc,columns=['class1','class2','class3','class4'])
                    writer =pd.ExcelWriter(save_csv_name)
                    df.to_excel(writer)
                    writer.save()
                    '''
                       
                   
            
                    precision = dict()
                    recall = dict()
                    average_precision = dict()
                    y_test = label_binarize(total_label, classes=[0, 1, 2,3])
                    n_classes = y_test.shape[1]	
                    y_score=total_scores
                    for i in range(n_classes):
                        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                            y_score[:, i])
                        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
                    
                    
                    average_precision["macro"] = average_precision_score(y_test, y_score,average="macro")
                    all_re = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
                    
                    # Then interpolate all ROC curves at this points
                    mean_pr = np.zeros_like(all_re)
                    for i in range(n_classes):
                        mean_pr += interp(all_re, recall[i], precision[i])
                        # Finally average it and compute AUC
                    mean_pr /= n_classes
                    
                    recall["macro"] = all_re
                    precision["macro"] = mean_pr
                    
                    print('Average precision score, macro-averaged over all classes: {0:0.2f}'
                          .format(average_precision["macro"]))
                    
                    # A ""micro"-average": quantifying score on all classes jointly
                    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                                    y_score.ravel())
                    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                                         average="micro")
                    print('Average precision score, macro-averaged over all classes: {0:0.2f}'
                          .format(average_precision["micro"]))
                 
                 
                    
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        
                    # A "macro-average": quantifying score on all classes jointly
                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                    
                        # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                        # Finally average it and compute AUC
                    mean_tpr /= n_classes
                    
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                   
                    roc_auc["macro"] = roc_auc_score(y_test, y_score, average="macro")
                    print('Average precision score, macro-averaged over all classes: {0:0.2f}'
                          .format(roc_auc["macro"]))
                    
                    # A "micro -average": quantifying score on all classes jointly
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),y_score.ravel())
                    roc_auc["micro"] = roc_auc_score(y_test, y_score, average="micro")
                    print('Average precision score, micro -averaged over all classes: {0:0.2f}'
                          .format(roc_auc["micro"]))
                    
                   
                    
                    save_csv_name='results/{}/output_pr.xlsx'.format(model_names[n])
                    writer =pd.ExcelWriter(save_csv_name)
                    

                    classes=['class1','class2','class3','class4',"macro",'micro']
                    index=[0,1,2,3,"macro",'micro']
                    j=0
                    for  i  in index:
                        name=classes[j]
                        df =pd.DataFrame( {'recall':np.array(recall[i]),'precision':np.array(precision[i]),
                                                       'average_precision':np.array(average_precision[i])})
                        df.to_excel(writer,sheet_name='{}'.format(name))
                        j=j+1
                    writer.save()
                    
                    
                    save_csv_name='results/{}/output_roc.xlsx'.format(model_names[n])
                    writer =pd.ExcelWriter(save_csv_name)

                    classes=['class1','class2','class3','class4',"macro",'micro']
                    index=[0,1,2,3,"macro",'micro']
                    j=0
                    for  i  in index:
                        name=classes[j]
                        df =pd.DataFrame( {'fpr':np.array(fpr[i]),'tpr':np.array(tpr[i]),
                                                       'roc_auc':np.array(roc_auc[i])})
                        df.to_excel(writer,sheet_name='{}'.format(name))
                        j=j+1
                    writer.save()
                    #print(precision)
                            
                   
