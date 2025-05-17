
import torch
import torch.nn.functional as F
import numpy as np
from IPython import embed

eps = 1e-8

class IQALoss(torch.nn.Module):
    def __init__(self, loss_type='linearity', alpha=[1, 0], beta=[.1, .1, 1], p=2, q=2, monotonicity_regularization=True, gamma=0.1, detach=False):
        super(IQALoss, self).__init__()
        self.loss_type = loss_type 
        self.alpha = alpha 
        self.beta = beta 
        
        self.p = p
        self.q = q
        
        self.monotonicity_regularization = monotonicity_regularization
        
        self.gamma = gamma
        
        self.detach = detach
    
    def forward(self, y_pred, y):
        '''
        y = y[0].view(-1, 1)
        loss = 0
        if self.beta[-1] > 0:
                loss += self.beta[-1] * self.loss_func(y_pred[-1], y)
        if self.beta[0] > 0:
            loss += self.beta[0] * self.loss_func(y_pred[0], y)
        if self.beta[1] > 0:
            loss += self.beta[1] * self.loss_func(y_pred[1], y)
        '''
        loss = self.loss_func(y_pred, y)

        return loss

    def loss_func(self, y_pred, y):
        if self.loss_type == 'mae':
            loss = F.l1_loss(y_pred, y)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'norm-in-norm':
            loss = norm_loss_with_normalization(y_pred, y, alpha=self.alpha, p=self.p, q=self.q, detach=self.detach)
        elif self.loss_type == 'min-max-norm':
            loss = norm_loss_with_min_max_normalization(y_pred, y, alpha=self.alpha, detach=self.detach)
        elif self.loss_type == 'mean-norm':
            loss = norm_loss_with_mean_normalization(y_pred, y, alpha=self.alpha, detach=self.detach)
        elif self.loss_type == 'scaling':
            loss = norm_loss_with_scaling(y_pred, y, alpha=self.alpha, p=self.p, detach=self.detach)
        else:
            loss = linearity_induced_loss(y_pred, y, self.alpha, detach=self.detach)
        if self.monotonicity_regularization:
            loss += self.gamma * monotonicity_regularization(y_pred, y, detach=self.detach)
        return loss



def monotonicity_regularization(y_pred, y, detach=False):
    """monotonicity regularization
    
    �������������ֵ i �� j������ʵֵ y[i] > y[j]����Ҫ��Ԥ��ֵ y_pred[i] > y_pred[j]��
    ͨ�� ReLU �Ͳ�ֵ�ķ��ţ�����һ��������������Υ�������Եĳ̶ȡ�
    """
    if y_pred.size(0) > 1:  
        ranking_loss = F.relu((y_pred-y_pred.t()) * torch.sign((y.t()-y)))
        scale = 1 + torch.max(ranking_loss.detach()) if detach else 1 + torch.max(ranking_loss)
        return torch.sum(ranking_loss) / y_pred.size(0) / (y_pred.size(0)-1) / scale
    else:
        return F.l1_loss(y_pred, y_pred.detach())  


def linearity_induced_loss(y_pred, y, alpha=[1, 1], detach=False):
    """linearity-induced loss, actually MSE loss with z-score normalization
    �����յ���ʧ��
    ���� z-score ��׼���� MSE ��ʧ����� Pearson ����Ժ����Իع��˼�롣
    ��׼������Ԥ��ֵ��Ŀ��ֵ���� z-score ��׼������������ֵ�ͷ����Ӱ�졣
    �����㣺
    loss0: ��׼����Ԥ��ֵ��Ŀ��ֵ�� MSE��������������� (1 - Pearson ���ϵ��)��
    loss1: ��Ԥ��ֵ��Ŀ��ֵ�����Թ�ϵ���� (1 - ����ϵ�� R2)��
    ������ʧ�������ּ�Ȩ�Ľ���� 
    """
    if y_pred.size(0) > 1:  
        sigma_hat, m_hat = torch.std_mean(y_pred.detach(), unbiased=False) if detach else torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + eps)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + eps)
        scale = 4
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / scale  
        if alpha[1] > 0:
            rho = torch.mean(y_pred * y)
            loss1 = F.mse_loss(rho * y_pred, y) / scale  
        
        
        
        
        
        
        
        
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  

def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm
    ���ڷ�����һ������ʧ��ͨ����Ԥ��ֵ��Ŀ��ֵ�������ֵ��һ�� (mean-centered normalization)����������
    ��һ����
    ��Ԥ��ֵ��Ŀ��ֵ�ľ�ֵ���㣬������ָ������ (p �� q) ��׼����
    �����㣺
    loss0: Ԥ��ֵ��Ŀ��ֵ�� Lp ������
    loss1: Ԥ��ֵ��Ŀ��ֵ���������ƶȵ�������
    """
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  
        
        y_pred = y_pred / (eps + normalization)  
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) 
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  
                err += eps 
            loss0 = torch.norm(err, p=p) / scale  
            loss0 = torch.pow(loss0, p) if exponent else loss0 
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  
            err = rho * y_pred - y
            if p < 1:  
                err += eps 
            loss1 = torch.norm(err, p=p) / scale  
            loss1 = torch.pow(loss1, p) if exponent else loss1 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  


def norm_loss_with_min_max_normalization(y_pred, y, alpha=[1, 1], detach=False):
    '''
    ���� Min-Max ��һ������ʧ��
    ʹ����Сֵ�����ֵ���й�һ�������������ݷ�Χ����ϴ�������

    ԭ����
    ��Ԥ��ֵ��Ŀ��ֵͨ�� (x - min) / (max - min) ��һ���� [0, 1] ��Χ��
    �����㣺
    loss0: ��׼ MSE ��ʧ��
    loss1: ����Ԥ��ֵ��Ŀ��ֵ�������ƶȵ������ MSE ��ʧ��
    '''
    if y_pred.size(0) > 1:  
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - m_hat) / (eps + M_hat - m_hat)  
        y = (y - torch.min(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y)
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  
            loss1 = F.mse_loss(rho * y_pred, y) 
            
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  


def norm_loss_with_mean_normalization(y_pred, y, alpha=[1, 1], detach=False):
    '''
    ���ھ�ֵ��һ������ʧ�����þ�ֵ��һ���ı��壬ͬʱ�������ֵ����Сֵ��

    ԭ����
        ��һ����
            ��Ԥ��ֵ��Ŀ��ֵ��׼��Ϊ mean(x)�����������ֵ����Сֵ������
    �����㣺
        ���� norm_loss_with_min_max_normalization��ʹ�� MSE ���������ƶȵ�����
    '''
    if y_pred.size(0) > 1:  
        mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - mean_hat) / (eps + M_hat - m_hat)  
        y = (y - torch.mean(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  


def norm_loss_with_scaling(y_pred, y, alpha=[1, 1], p=2, detach=False):
    if y_pred.size(0) > 1:  
        normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p) 
        y_pred = y_pred / (eps + normalization)  
        y = y / (eps + torch.norm(y, p=p))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  