# -*- coding: iso-8859-1 -*-  
import torch
import numpy as np
from torch.optim import lr_scheduler
import clip
from clip_main import model_clipmoe_5experts
import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc_3k, _preprocess2, _preprocess3, convert_models_to_fp32, get_logger,log_and_print
import torch.nn.functional as F
from itertools import product
import os
import tqdm
import EnhancedCLIPNetworkv3
from IQAloss import IQALoss

from IPython import embed


checkpoint_dir = 'result/all_MoE_align_5experts_step_test_patch_nndloss'
# checkpoint_dir = 'result/all_MoE_align_nonnd_nostep_five_experts_nndloss_bs=32_trainPatch=8_testPatch=32_step'
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
num_epoch = 60
bs = 32
train_patch = 4
test_patch = 32
num_workers = 16

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

opt = 0

# 
def freeze_model(opt):
    # ÍøÂç½á¹¹ÄÇ±ßÅªÁË
    # model.logit_scale.requires_grad = False  
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in model.clip_model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.clip_model.transformer.parameters():
            p.requires_grad = False
        model.clip_model.positional_embedding.requires_grad = False
        model.clip_model.text_projection.requires_grad = False
        for p in model.clip_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.clip_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.clip_model.parameters():
            p.requires_grad =False
    elif opt == 4:
        for p in model.clip_model.parameters():  
            p.requires_grad = True  
    elif opt == 5: # freeze text_adv encoder
        for p in model.adv_clip.token_embedding_adv.parameters():
            p.requires_grad = False
        for p in model.adv_clip.additional_text_encoder.parameters():
            p.requires_grad = False
        model.adv_clip.positional_embedding_adv.requires_grad = False
        model.adv_clip.text_projection_adv.requires_grad = False
        for p in model.adv_clip.ln_final_adv.parameters():
            p.requires_grad = False
        




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

            input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)  # shape:[80,77]
            
            texts_mos = [f"A photo of {c} quality" for c in qualitys_p1] * len(prompt)
            input_texts_mos = torch.cat([clip.tokenize(c,truncate=True) for c in texts_mos]).to(device)  # shape:[80,77]

            optimizer.zero_grad()
            # print(f'step:{step},x:{x}')
            
            # logits_quality, logits_quality_adv, moe_loss = model(x, input_texts, input_texts_mos)
            # print(f'output:{logits_quality}')
            logits_quality,  logits_quality_col, logits_quality_adv, moe_loss = model(x, input_texts, input_texts_mos)
            

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            logits_quality = ((logits_quality -1) / 4) *5      
            
            logits_quality_col = 1 * logits_quality_col[:, 0] + 2 * logits_quality_col[:, 1] + 3 * logits_quality_col[:, 2] + \
                                4 * logits_quality_col[:, 3] + 5 * logits_quality_col[:, 4]
            logits_quality_col = ((logits_quality_col -1) / 4) *5  
            
            logits_quality_adv = 1 * logits_quality_adv[:, 0] + 2 * logits_quality_adv[:, 1] + 3 * logits_quality_adv[:, 2] + \
                                4 * logits_quality_adv[:, 3] + 5 * logits_quality_adv[:, 4]
            logits_quality_adv = ((logits_quality_adv -1) / 4) *5  
            
            loss_func = IQALoss(loss_type='norm-in-norm', alpha = [1,1], p=1, q=2 ,gamma=0.1)
            
            alpha = 0.1
            beta = 0.01
            
            
            total_loss = loss_func(logits_quality.unsqueeze(1), galign.detach().unsqueeze(1)) + loss_func(logits_quality_col.unsqueeze(1), gmos.detach().unsqueeze(1)) + loss_func(logits_quality_adv.unsqueeze(1), gmos.detach().unsqueeze(1)) + moe_loss
           
            
            align_loss = loss_func(logits_quality.unsqueeze(1), galign.detach().unsqueeze(1))
            col_loss =  loss_func(logits_quality_col.unsqueeze(1), gmos.detach().unsqueeze(1))
            adv_loss =  loss_func(logits_quality_adv.unsqueeze(1), gmos.detach().unsqueeze(1))
            
            '''
            total_loss = loss_m3(logits_quality, galign.detach()).mean() + loss_m3(logits_quality_col, gmos.detach()).mean() + loss_m3(logits_quality_adv, gmos.detach()).mean() + moe_loss  # Ñµalign
            align_loss =  loss_m3(logits_quality, galign.detach()).mean() 
            col_loss =   loss_m3(logits_quality_col, gmos.detach()).mean()
            adv_loss =  loss_m3(logits_quality_adv, gmos.detach()).mean()
            '''

            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                model_clipmoe_5experts.convert_weights(model)


            running_loss += total_loss.data.item()
                 
            avg_loss = running_loss / (step + 1)
            # print(avg_loss)
            loop.set_description('Epoch:{}  Loss:{:.4f}'.format(epoch, avg_loss))
        
        log_and_print(base_logger, 'Epoch:{} Average Loss: {:.4f}'.format(epoch, avg_loss))  
        
        log_and_print(base_logger, f'align loss: {align_loss}')
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

        x, gmos, prompt, image_name = sample_batched['I'], sample_batched['align'],  sample_batched['prompt'], sample_batched['image_name']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()
        
        
        # ÑµÁ·align
        texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
        input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)
        
        
        '''
        # ÑµÁ·mos  
        texts_mos = [f"A photo of {c} quality" for c in qualitys_p1] * len(prompt)
        input_texts_mos = torch.cat([clip.tokenize(c,truncate=True) for c in texts_mos]).to(device)  # shape:[80,77]
        '''
        
        with torch.no_grad():
            logits_quality,_,_,_ = model(x, input_texts)
        
        
        '''
        # ÑµÁ·mos
        texts_mos = [f"A photo of {c} quality" for c in qualitys_p1] * len(prompt)
        input_texts_mos = torch.cat([clip.tokenize(c,truncate=True) for c in texts_mos]).to(device)  # shape:[80,77]

        with torch.no_grad():
            logits_quality,_ = model(x, input_texts_mos)
        '''

        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        
        quality_preds = ((quality_preds -1) / 4) *5 
        q_hat = q_hat + quality_preds.cpu().tolist()
        


    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    #¼ÆËãPLCC
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}'.format(srcc,plcc)
    # print_text = dataset + ' ' + phase + ' finished'
    log_and_print(base_logger,print_text)

    return  (srcc+plcc)/2, srcc, plcc




best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
for session in range(0,10):
    # model =  EnhancedCLIPNetworkv3.CLIPMoE(top_k=6,num_experts=10,dropout=0.1, moe_layers=1)
    # model =  EnhancedCLIPNetworkv3.CLIPMoE(top_k=2,num_experts=4,dropout=0.1, moe_layers=6)
    model =  EnhancedCLIPNetworkv3.CLIPMoE(moe_layers=12, topk=1)
    freeze_model(0)  
    model.to(device) 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    best_result = {'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
    best_epoch = {'quality': 0}


    aigc_train_csv = os.path.join('../Database/AGIQA-3K', str(session+1), 'train.csv')
    aigc_val_csv = os.path.join('../Database/AGIQA-3K', str(session+1), 'val.csv')

    aigc_set = '../data/AGIQA-3K'
    # 8ºÍ15ÊÇÇÐ·ÖpatchµÄÊýÄ¿,train_patch
    aigc_train_loader = set_dataset_aigc_3k(aigc_train_csv, bs, aigc_set, num_workers, preprocess3, train_patch, False)
    aigc_val_loader = set_dataset_aigc_3k(aigc_val_csv, bs , aigc_set, num_workers, preprocess2, test_patch, True)

    train_loaders = [aigc_train_loader]
    freeze_model(0)  
    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch = train(model, best_result, best_epoch)
        scheduler.step()
        
        # ÔÚÃ¿¸ö epoch ºó²é¿´ÌØ¶¨²ãµÄÈ¨ÖØ  
        #if epoch % 1 == 0:  # Ã¿¸ö epoch ²é¿´Ò»´Î  
            # ·ÃÎÊµÚÒ»¸ö ResidualAttentionBlock µÄ attn ²ã  
            # layer_weights = model.visual.transformer.resblocks[0].attn.out_proj.weight.data  
            
            # ¼ÆËã¾ùÖµºÍ±ê×¼²î  
            #weights_mean = layer_weights.mean().item()  
            #weights_std = layer_weights.std().item()  
            
            #log_and_print(base_logger, 'Epoch: {}, Layer Weights Mean: {:.4f}, Std: {:.4f}'.format(epoch + 1, weights_mean, weights_std)) 
            #print(f'Epoch: {epoch + 1}, c_fc Weights (first 10): {layer_weights[:10]}')
         
        log_and_print(base_logger,'...............current quality best, session:{}...............'.format(session + 1))
        log_and_print(base_logger,'best quality epoch:{}'.format(best_epoch['quality']))
        log_and_print(base_logger,'best quality result:{}, srcc:{}, plcc:{}'.format(best_result['quality'], best_result['srcc'], best_result['plcc']))

    best_result_list.append(best_result)


avg_srcc = 0
avg_plcc = 0
for i in range(10):
    # avg_srcc += best_result_list[i+1]['srcc']
    # avg_plcc += best_result_list[i+1]['plcc']
    avg_srcc += best_result_list[i]['srcc']
    avg_plcc += best_result_list[i]['plcc']


avg_srcc = avg_srcc / 10
avg_plcc = avg_plcc / 10
log_and_print(base_logger,'all_finished,average srcc:{}, plcc:{}'.format(avg_srcc, avg_plcc))
