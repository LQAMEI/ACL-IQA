
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import clip
import tools

class FeatureExtractor(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(FeatureExtractor, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_conv1', nn.Conv2d(input_channel, 64, kernel_size=3))
        feature.add_module('f_bn1', nn.BatchNorm2d(64))
        feature.add_module('f_pool1', nn.MaxPool2d(2))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_conv2', nn.Conv2d(64, output_channel, kernel_size=3))
        feature.add_module('f_bn2', nn.BatchNorm2d(output_channel))
        feature.add_module('f_drop1', nn.Dropout2d())
        feature.add_module('f_pool2', nn.MaxPool2d(2))
        feature.add_module('f_relu2', nn.ReLU(True))
        self.feature = feature

    def forward(self, x):
        return self.feature(x)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha): 
        ctx.alpha = alpha
        return x.view_as(x) 

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_class=5):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        
        self.dis2 = nn.Linear(hidden_dim, num_class) 

    def forward(self, x):
        x = F.relu(self.dis1(x))
        
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x

class DANN(nn.Module):
    def __init__(self, device, num_class=5):
        super(DANN, self).__init__()
        self.device = device
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        
        
        self.domain_classifier = Discriminator(input_dim = 512, hidden_dim = 64, num_class=num_class)
        
    def forward(self, x, text, source=None, alpha=1):
        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        
        
        image_features = self.clip.encode_image(x)
        text_features =  self.clip.encode_text(text)
        
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True) 
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale =  self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        
        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        
        logits_per_image = torch.cat((logits_per_image[:,:num_patch-1,:].mean(1).unsqueeze(1), logits_per_image[:,-1,:].unsqueeze(1)), 1)

        logits_per_image_p = logits_per_image[0,:,:5].unsqueeze(0) 
        if batch_size > 1:
            for i in range(1, batch_size):
                logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,:,i*5:(i+1)*5].unsqueeze(0)), 0)

        logits_per_image_p = F.softmax(logits_per_image_p, dim=2) 

        
        loss_adv = None
        if source is not None:
            loss_fn = nn.CrossEntropyLoss() 
            image_features = image_features.view(batch_size, num_patch, -1)
            
            image_features = torch.sum(image_features, dim=1) / image_features.shape[1]
            domain_label = torch.tensor(source, dtype=torch.long).to(self.device)
            
            domain_x = ReverseLayerF.apply(image_features, alpha)
            domain_pred = self.domain_classifier(domain_x) 
            loss_adv = loss_fn(domain_pred, domain_label.long())

        return logits_per_image_p.mean(1), loss_adv
    
    def get_clip(self):
        return self.clip

    def get_preprocess(self):
        return self.preprocess
    
    def freeze_clip(self, opt):
        self.clip.logit_scale.requires_grad = False
        if opt == 0: 
            return
        elif opt == 1: 
            for p in self.clip.token_embedding.parameters():
                p.requires_grad = False
            for p in self.clip.transformer.parameters():
                p.requires_grad = False
            self.clip.positional_embedding.requires_grad = False
            self.clip.text_projection.requires_grad = False
            for p in self.clip.ln_final.parameters():
                p.requires_grad = False
        elif opt == 2: 
            for p in self.clip.visual.parameters():
                p.requires_grad = False
        elif opt == 3:
            for p in self.clip.parameters():
                p.requires_grad =False


class DANN_DA(nn.Module):
    def __init__(self, device, input_dim=512, hidden_dim=64):
        super(DANN_DA, self).__init__()
        self.device = device
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.domain_classifier = Discriminator(input_dim, hidden_dim)
        
    def forward(self, x, text, source=True, alpha=1):
        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        image_features = self.clip.encode_image(x)
        text_features =  self.clip.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True) 
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale =  self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        logits_per_image = torch.cat((logits_per_image[:,:num_patch-1,:].mean(1).unsqueeze(1), logits_per_image[:,-1,:].unsqueeze(1)), 1)

        logits_per_image_p = logits_per_image[0,:,:5].unsqueeze(0)
        if batch_size > 1:
            for i in range(1, batch_size):
                logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,:,i*5:(i+1)*5].unsqueeze(0)), 0)

        logits_per_image_p = F.softmax(logits_per_image_p, dim=2) 

        loss_fn = nn.BCEWithLogitsLoss()
        image_features = image_features.view(batch_size, num_patch, -1)
        image_features = torch.sum(image_features, dim=1) / image_features.shape[1]
        
        if source:
            domain_label = torch.ones(image_features[0]).long().to(self.device)
        else:
            domain_label = torch.zeros(image_features[0]).long().to(self.device)
        
        domain_x = ReverseLayerF.apply(image_features, alpha)
        domain_pred = self.domain_classifier(domain_x)
        domain_pred = domain_pred.squeeze()  
        loss_adv = loss_fn(domain_pred, domain_label.long())

        return logits_per_image_p.mean(1), loss_adv
    
    def get_clip(self):
        return self.clip

    def get_preprocess(self):
        return self.preprocess
    
    def freeze_clip(self, opt):
        self.clip.logit_scale.requires_grad = False
        if opt == 0: 
            return
        elif opt == 1: 
            for p in self.clip.token_embedding.parameters():
                p.requires_grad = False
            for p in self.clip.transformer.parameters():
                p.requires_grad = False
            self.clip.positional_embedding.requires_grad = False
            self.clip.text_projection.requires_grad = False
            for p in self.clip.ln_final.parameters():
                p.requires_grad = False
        elif opt == 2: 
            for p in self.clip.visual.parameters():
                p.requires_grad = False
        elif opt == 3:
            for p in self.clip.parameters():
                p.requires_grad =False

def get_adversarial_result(x, source, input_channel=3, output_channel=50, alpha=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loss_fn = nn.CrossEntropyLoss()  
    extractor = FeatureExtractor(input_channel, output_channel).to(device)
    x = extractor(x)
    input_dim = x.size(1) * x.size(2) * x.size(3)
    x = x.view(x.size(0),-1)
    x = x.to(device)
    
    domain_classifier = Discriminator(input_dim).to(device)
    domain_label = torch.tensor(source, dtype=torch.long).to(device)
    
    x = ReverseLayerF.apply(x, alpha)
    domain_pred = domain_classifier(x)
    domain_pred = domain_pred.squeeze()  
    loss_adv = loss_fn(domain_pred, domain_label.float())
    return loss_adv
