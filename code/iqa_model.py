 # -*- coding: iso-8859-1 -*-  
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from clip_main import acl_iqa

from IPython import embed

class ACLIQA(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32",device="cuda", topk=1):
        super(ACLIQA, self).__init__()           
        self.model, _ = acl_iqa.loadACLIQA(clip_model_name, device=device, topk=topk)
        
    def get_patch_image_p_mean(self,logits_per_image,batch_size,num_patch):
        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        logits_per_image = torch.cat((logits_per_image[:,:num_patch-1,:].mean(1).unsqueeze(1), logits_per_image[:,-1,:].unsqueeze(1)), 1)
    
        logits_per_image_p = logits_per_image[0,:,:5].unsqueeze(0)

        if batch_size > 1:
            for i in range(1, batch_size):
                logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,:,i*5:(i+1)*5].unsqueeze(0)), 0)

        logits_per_image_p = F.softmax(logits_per_image_p, dim=2)
        logits_per_image_p_mean = logits_per_image_p.mean(1) # [16,5]
        return logits_per_image_p_mean
        
        
    def forward(self, x, text1, col_text, adv_text, text2=None):
        batch_size = x.size(0)  
        num_patch = x.size(1) 

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        logits_per_image,logits_per_image_col, logits_per_image_adv,moe_loss = self.model(x, text1, col_text, adv_text, text2)  # logits_per_image torch.Size([144, 80])

        logits_per_image_p_mean = self.get_patch_image_p_mean(logits_per_image,batch_size,num_patch)
        
        logits_per_image_col_p_mean = None
        logits_per_image_adv_p_mean = None

        if text2 is not None and logits_per_image_col is not None: 
            if logits_per_image_col is not None:
                logits_per_image_col_p_mean = self.get_patch_image_p_mean(logits_per_image_col,batch_size,num_patch)
            if logits_per_image_adv is not None:
                logits_per_image_adv_p_mean = self.get_patch_image_p_mean(logits_per_image_adv,batch_size,num_patch)
            
        return logits_per_image_p_mean, logits_per_image_col_p_mean, logits_per_image_adv_p_mean, moe_loss



class ConvMoE(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32",device="cuda", topk=1):
        super(ConvMoE, self).__init__()           
        self.model, _ = acl_iqa.loadMoE(clip_model_name, device=device, topk=topk)

        
    def get_patch_image_p_mean(self,logits_per_image,batch_size,num_patch):
        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        logits_per_image = torch.cat((logits_per_image[:,:num_patch-1,:].mean(1).unsqueeze(1), logits_per_image[:,-1,:].unsqueeze(1)), 1)
    
        logits_per_image_p = logits_per_image[0,:,:5].unsqueeze(0)

        if batch_size > 1:
            for i in range(1, batch_size):
                logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,:,i*5:(i+1)*5].unsqueeze(0)), 0)

        logits_per_image_p = F.softmax(logits_per_image_p, dim=2)
        logits_per_image_p_mean = logits_per_image_p.mean(1) # [16,5]
        return logits_per_image_p_mean
        
        
    def forward(self, x, text1, sema):
        batch_size = x.size(0) 
        num_patch = x.size(1) 
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        logits_per_image,  moe_loss = self.model(x, text1, sema)  
        logits_per_image_p_mean = self.get_patch_image_p_mean(logits_per_image,batch_size,num_patch)
        
        return logits_per_image_p_mean,  moe_loss