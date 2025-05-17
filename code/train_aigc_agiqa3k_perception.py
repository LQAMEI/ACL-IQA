# -*- coding: iso-8859-1 -*-  
import torch
import numpy as np
from torch.optim import lr_scheduler

from clip_main import model as model_moe
from clip_main import acl_iqa

import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc_3k, _preprocess2, _preprocess3, convert_models_to_fp32, get_logger,log_and_print
import torch.nn.functional as F
from itertools import product
import os
import tqdm
import iqa_model
from IPython import embed


checkpoint_dir = 'result/3k_qual_100epoch'
os.makedirs(checkpoint_dir,exist_ok = True)

qualitys_p = ['badly', 'poorly', 'fairly', 'well', 'perfectly']
qualitys_p1 = ["bad", "poor", "fair", "good", "perfect"]

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 100
bs = 16
train_patch = 8
test_patch = 15

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

def freeze_model(opt):
    model.model.logit_scale.requires_grad = False
    if opt == 0: 
        return
    elif opt == 1: # freeze text encoder
        for p in model.model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.model.encoder.text_resblocks.parameters():
            p.requires_grad = False
        model.model.text_positional_embedding.requires_grad = False
        model.model.text_projection.requires_grad = False
        for p in model.model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.model.encoder.vision_resblocks.parameters():
            p.requires_grad = False
        for p in model.model.conv1.parameters():  
            p.requires_grad = False  
        model.model.class_embedding.requires_grad = False  
        model.model.positional_embedding.requires_grad = False  
 
        for p in model.model.ln_pre.parameters():
            p.requires_grad = False
        for p in model.model.ln_post_col.parameters():
            p.requires_grad = False            
        for p in model.model.ln_post_adv.parameters():
            p.requires_grad = False     
        model.model.proj_col.requires_grad = False  
        model.model.proj_adv.requires_grad = False                 
    
    elif opt == 3:
        for p in model.model.parameters():
            p.requires_grad =False



def train(model, best_result, best_epoch):
    with torch.cuda.amp.autocast(enabled=True):
    
        model.train()
        running_loss = 0

        loader = train_loaders[0]

        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
            scheduler.step()
           # print(optimizer.state_dict()['param_groups'][0]['lr'])

        avg_loss = 0
        step = -1
        loop = tqdm.tqdm(loader, desc='Epoch:{}'.format(epoch))
        for sample_batched in loop:
            step += 1
            x, gmos, galign, prompt, image_name = sample_batched['I'], sample_batched['mos'], sample_batched['align'],  sample_batched['prompt'], sample_batched['image_name']
            x = x.to(device)
            gmos = gmos.to(device)
            galign = galign.to(device)
            texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
            input_texts = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in texts]).to(device)  # shape:[80,77]
            
            input_texts_se = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in prompt]).to(device)  # shape:[16,77]
            
            col_texts = [f"In a photo like '{p}', using alignment to assist in evaluating image quality" for p in prompt]
            input_col_texts = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in col_texts]).to(device)  # shape:[16,77]
            
            adv_texts = [f"In a photo like '{p}', excluding consideration of alignment when evaluating image quality" for p in prompt]
            input_adv_texts = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in adv_texts]).to(device)  # shape:[16,77]
            
            texts_qual = [f"A photo of {c} quality" for c in qualitys_p1] * len(prompt)
            input_texts_qual = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in texts_qual]).to(device)  # shape:[80,77]
            
            optimizer.zero_grad()

            logits_quality, logits_quality_col, logits_quality_adv, moe_loss = model(x, input_texts_qual, input_col_texts, input_adv_texts, input_texts)
            

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            logits_quality = ((logits_quality -1) / 4) *5      
            
            total_loss = loss_m3(logits_quality, gmos.detach()).mean() + moe_loss
            
            if logits_quality_col is not None:
                logits_quality_col = 1 * logits_quality_col[:, 0] + 2 * logits_quality_col[:, 1] + 3 * logits_quality_col[:, 2] + \
                                    4 * logits_quality_col[:, 3] + 5 * logits_quality_col[:, 4]
                logits_quality_col = ((logits_quality_col -1) / 4) *5
                total_loss += loss_m3(logits_quality_col, galign.detach()).mean()
            
            
            if logits_quality_adv is not None:
                logits_quality_adv = 1 * logits_quality_adv[:, 0] + 2 * logits_quality_adv[:, 1] + 3 * logits_quality_adv[:, 2] + \
                        4 * logits_quality_adv[:, 3] + 5 * logits_quality_adv[:, 4]
                logits_quality_adv = ((logits_quality_adv -1) / 4) *5
                total_loss += loss_m3(logits_quality_adv, galign.detach()).mean()
            
            qual_loss = loss_m3(logits_quality, gmos.detach()).mean()
            col_loss =  loss_m3(logits_quality_col, galign.detach()).mean()
            adv_loss =  loss_m3(logits_quality_adv, galign.detach()).mean()
            moe_loss = moe_loss
            
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                model_moe.convert_weights(model)


            running_loss += total_loss.data.item()
                 
            avg_loss = running_loss / (step + 1)
            # print(avg_loss)
            loop.set_description('Epoch:{}  Loss:{:.4f}'.format(epoch, avg_loss))
        
        log_and_print(base_logger, 'Epoch:{} Average Loss: {:.4f}'.format(epoch, avg_loss))  
        
        log_and_print(base_logger, f'qual loss: {qual_loss}')
        log_and_print(base_logger, f'col loss: {col_loss}')
        log_and_print(base_logger, f'adv loss: {adv_loss}')
        log_and_print(base_logger, f'moe loss: {moe_loss}')
        

        
        if (epoch >= 0):
            avg_score,srcc,plcc = eval(aigc_val_loader, phase='val', dataset='live')

            if avg_score > best_result['quality']:
                log_and_print(base_logger,'**********New quality best!**********')
                best_epoch['quality'] = epoch
                best_result['quality'] = avg_score
                best_result['srcc'] = srcc
                best_result['plcc'] = plcc
                dir = os.path.join(checkpoint_dir, str(session + 1))
                os.makedirs(dir,exist_ok = True)
                ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'quality_best_ckpt.pt')
                
                torch.save({
                    # 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'all_results': all_result
                }, ckpt_name)  # just change to your preferred folder/filename

        return best_result, best_epoch


def eval(loader, phase, dataset):
    model.eval()

    q_mos = []
    q_hat = []

    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):

        x, gmos, prompt, image_name = sample_batched['I'], sample_batched['mos'],  sample_batched['prompt'], sample_batched['image_name']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()
        
        texts_qual = [f"A photo of {c} quality" for c in qualitys_p1] * len(prompt)
        input_texts_qual = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in texts_qual]).to(device)  # shape:[80,77]

        col_texts = [f"In a photo like '{p}', using alignment to assist in evaluating image quality" for p in prompt]
        input_col_texts = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in col_texts]).to(device)  # shape:[16,77]
        
        adv_texts = [f"In a photo like '{p}', excluding consideration of alignment when evaluating image quality" for p in prompt]
        input_adv_texts = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in adv_texts]).to(device)  # shape:[16,77]
                  
        input_texts_se = torch.cat([acl_iqa.tokenize(c,truncate=True) for c in prompt]).to(device)  # shape:[16,77]

        
        with torch.no_grad():
            logits_quality,_,_,_ = model(x, input_texts_qual, input_col_texts, input_adv_texts)
        
        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        
        quality_preds = ((quality_preds -1) / 4) *5 
        q_hat = q_hat + quality_preds.cpu().tolist()
        


    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    #����PLCC
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}'.format(srcc,plcc)
    # print_text = dataset + ' ' + phase + ' finished'
    log_and_print(base_logger,print_text)

    return  (srcc+plcc)/2, srcc, plcc


num_workers = 16
best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
for session in range(0,10):
    model =  iqa_model.ACLIQA(topk=3)

    model.to(device) 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0
    
    freeze_model(0)

    best_result = {'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
    best_epoch = {'quality': 0}


    aigc_train_csv = os.path.join('../Database/AGIQA-3K', str(session+1), 'train.csv')
    aigc_val_csv = os.path.join('../Database/AGIQA-3K', str(session+1), 'val.csv')

    aigc_set = '../data/AGIQA-3K'
    aigc_train_loader = set_dataset_aigc_3k(aigc_train_csv, bs, aigc_set, num_workers, preprocess3, train_patch, False)
    aigc_val_loader = set_dataset_aigc_3k(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)

    train_loaders = [aigc_train_loader]
 
    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch = train(model, best_result, best_epoch)
        scheduler.step()
        
        log_and_print(base_logger,'...............current quality best, session:{}...............'.format(session + 1))
        log_and_print(base_logger,'best quality epoch:{}'.format(best_epoch['quality']))
        log_and_print(base_logger,'best quality result:{}, srcc:{}, plcc:{}'.format(best_result['quality'], best_result['srcc'], best_result['plcc']))

    best_result_list.append(best_result)


avg_srcc = 0
avg_plcc = 0
for i in range(10):
    avg_srcc += best_result_list[i]['srcc']
    avg_plcc += best_result_list[i]['plcc']


avg_srcc = avg_srcc / 10
avg_plcc = avg_plcc / 10
log_and_print(base_logger,'all_finished,average srcc:{}, plcc:{}'.format(avg_srcc, avg_plcc))
