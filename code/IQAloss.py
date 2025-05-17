# -*- coding: iso-8859-1 -*-  
import torch
import torch.nn.functional as F
import numpy as np
from IPython import embed

eps = 1e-8

class IQALoss(torch.nn.Module):
    def __init__(self, loss_type='linearity', alpha=[1, 0], beta=[.1, .1, 1], p=2, q=2, monotonicity_regularization=True, gamma=0.1, detach=False):
        super(IQALoss, self).__init__()
        self.loss_type = loss_type # loss_type: ËðÊ§ÀàÐÍ£¬¿ÉÑ¡Öµ°üÀ¨ linearity¡¢mae¡¢mse µÈ¡£
        self.alpha = alpha # ¿ØÖÆÌØ¶¨ËðÊ§·ÖÁ¿µÄÈ¨ÖØ¡£
        self.beta = beta # Õë¶Ô²»Í¬½×¶Î»ò·ÖÁ¿µÄËðÊ§È¨ÖØ¡£
        # ÓÃÓÚ¶¨Òå·¶ÊýµÄÃÝ´Î
        self.p = p
        self.q = q
        # ÊÇ·ñÆôÓÃµ¥µ÷ÐÔÕýÔò»¯¡£
        self.monotonicity_regularization = monotonicity_regularization
        # µ¥µ÷ÐÔÕýÔò»¯µÄÈ¨ÖØ
        self.gamma = gamma
        # detach: ¿ØÖÆÊÇ·ñÔÚÄ³Ð©¼ÆËãÖÐÊ¹ÓÃ detach() ¼õÉÙÌÝ¶ÈÁ÷¡£
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
    # µ¥µ÷ÐÔÕýÔò»¯£ºÓÃÓÚ¹ÄÀøÔ¤²âÖµÓëÄ¿±êÖµÖ®¼äµÄµ¥µ÷ÐÔ¹ØÏµ£¬ÌØ±ðÊÊÓÃÓÚÅÅÐòÏà¹ØÈÎÎñ¡£
    Èç¹ûÑù±¾ÓÐÁ½¸öÖµ i ºÍ j£¬ÇÒÕæÊµÖµ y[i] > y[j]£¬ÔòÒªÇóÔ¤²âÖµ y_pred[i] > y_pred[j]¡£
    Í¨¹ý ReLU ºÍ²îÖµµÄ·ûºÅ£¬¹¹½¨Ò»¸öÕýÔòÏîÀ´¶ÈÁ¿Î¥·´µ¥µ÷ÐÔµÄ³Ì¶È¡£
    """
    if y_pred.size(0) > 1:  #
        ranking_loss = F.relu((y_pred-y_pred.t()) * torch.sign((y.t()-y)))
        scale = 1 + torch.max(ranking_loss.detach()) if detach else 1 + torch.max(ranking_loss)
        return torch.sum(ranking_loss) / y_pred.size(0) / (y_pred.size(0)-1) / scale
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def linearity_induced_loss(y_pred, y, alpha=[1, 1], detach=False):
    """linearity-induced loss, actually MSE loss with z-score normalization
    ÏßÐÔÓÕµ¼ËðÊ§£º
    »ùÓÚ z-score ±ê×¼»¯µÄ MSE ËðÊ§£¬½áºÏ Pearson Ïà¹ØÐÔºÍÏßÐÔ»Ø¹éµÄË¼Ïë¡£
    ±ê×¼»¯£º¶ÔÔ¤²âÖµºÍÄ¿±êÖµ½øÐÐ z-score ±ê×¼»¯£¬ÒÔÏû³ý¾ùÖµºÍ·½²îµÄÓ°Ïì¡£
    Îó²î¼ÆËã£º
    loss0: ±ê×¼»¯ºóÔ¤²âÖµºÍÄ¿±êÖµµÄ MSE£¬ºâÁ¿ÏßÐÔÏà¹ØÐÔ (1 - Pearson Ïà¹ØÏµÊý)¡£
    loss1: ÓÃÔ¤²âÖµÓëÄ¿±êÖµµÄÏßÐÔ¹ØÏµºâÁ¿ (1 - ¾ö¶¨ÏµÊý R2)¡£
    ×îÖÕËðÊ§ÊÇÁ½²¿·Ö¼ÓÈ¨µÄ½á¹û¡£ 
    """
    if y_pred.size(0) > 1:  # z-score normalization: (x-m(x))/sigma(x).
        sigma_hat, m_hat = torch.std_mean(y_pred.detach(), unbiased=False) if detach else torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + eps)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + eps)
        scale = 4
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / scale  # ~ 1 - rho, rho is PLCC
        if alpha[1] > 0:
            rho = torch.mean(y_pred * y)
            loss1 = F.mse_loss(rho * y_pred, y) / scale  # 1 - rho ** 2 = 1 - R^2, R^2 is Coefficient of determination
        # loss0 =  (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2
        # yp = y_pred.detach() if detach else y_pred
        # ones = torch.ones_like(yp.detach())
        # yp1 = torch.cat((yp, ones), dim=1)
        # h = torch.mm(torch.inverse(torch.mm(yp1.t(), yp1)), torch.mm(yp1.t(), y))
        # err = torch.pow(torch.mm(torch.cat((y_pred, ones), dim=1), h) - y, 2)  #
        # normalization = 1 + torch.max(err.detach()) if detach else 1 + torch.max(err)
        # loss1 = torch.mean(err) / normalization
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm
    »ùÓÚ·¶Êý¹éÒ»»¯µÄËðÊ§£ºÍ¨¹ý¶ÔÔ¤²âÖµºÍÄ¿±êÖµ½øÐÐÁã¾ùÖµ¹éÒ»»¯ (mean-centered normalization)£¬¼ÆËãÆäÎó²î¡£
    ¹éÒ»»¯£º
    ½«Ô¤²âÖµºÍÄ¿±êÖµµÄ¾ùÖµ¹éÁã£¬²¢°´ÕÕÖ¸¶¨·¶Êý (p ºÍ q) ±ê×¼»¯¡£
    Îó²î¼ÆËã£º
    loss0: Ô¤²âÖµºÍÄ¿±êÖµµÄ Lp ·¶ÊýÎó²î¡£
    loss1: Ô¤²âÖµÓëÄ¿±êÖµµÄÓàÏÒÏàËÆ¶Èµ÷ÕûºóÎó²î¡£
    """
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  # very important!!
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
        # print('bhat = {}'.format(normalization.item()))
        y_pred = y_pred / (eps + normalization)  # very important!
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
            loss0 = torch.pow(loss0, p) if exponent else loss0 #
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #  
            err = rho * y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss1 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1 #  #  
        # by = normalization.detach()
        # e0 = err.detach().view(-1)
        # ones = torch.ones_like(e0)
        # yhat = y_pred.detach().view(-1)
        # g0 = torch.norm(e0, p=p) / torch.pow(torch.norm(e0, p=p) + eps, p) * torch.pow(torch.abs(e0), p-1) * e0 / (torch.abs(e0) + eps)
        # ga = -ones / N * torch.dot(g0, ones)
        # gg0 = torch.dot(g0, g0)
        # gga = torch.dot(g0+ga, g0+ga)
        # print("by: {} without a and b: {} with a: {}".format(normalization, gg0, gga))
        # gb = -torch.pow(torch.abs(yhat), q-1) * yhat / (torch.abs(yhat) + eps) * torch.dot(g0, yhat)
        # gab = torch.dot(ones, torch.pow(torch.abs(yhat), q-1) * yhat / (torch.abs(yhat) + eps)) / N * torch.dot(g0, yhat)
        # ggb = torch.dot(g0+gb, g0+gb)
        # ggab = torch.dot(g0+ga+gb+gab, g0+ga+gb+gab)
        # print("by: {} without a and b: {} with a: {} with b: {} with a and b: {}".format(normalization, gg0, gga, ggb, ggab))
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_min_max_normalization(y_pred, y, alpha=[1, 1], detach=False):
    '''
    »ùÓÚ Min-Max ¹éÒ»»¯µÄËðÊ§£º
    Ê¹ÓÃ×îÐ¡ÖµºÍ×î´óÖµ½øÐÐ¹éÒ»»¯£¬ÊÊÓÃÓÚÊý¾Ý·¶Î§²îÒì½Ï´óµÄÇé¿ö¡£

    Ô­Àí£º
    ½«Ô¤²âÖµºÍÄ¿±êÖµÍ¨¹ý (x - min) / (max - min) ¹éÒ»»¯µ½ [0, 1] ·¶Î§¡£
    Îó²î¼ÆËã£º
    loss0: ±ê×¼ MSE ËðÊ§¡£
    loss1: »ùÓÚÔ¤²âÖµºÍÄ¿±êÖµÓàÏÒÏàËÆ¶Èµ÷ÕûºóµÄ MSE ËðÊ§¡£
    '''
    if y_pred.size(0) > 1:  
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - m_hat) / (eps + M_hat - m_hat)  # min-max normalization
        y = (y - torch.min(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y)
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) 
            
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_mean_normalization(y_pred, y, alpha=[1, 1], detach=False):
    '''
    »ùÓÚ¾ùÖµ¹éÒ»»¯µÄËðÊ§£ºÀûÓÃ¾ùÖµ¹éÒ»»¯µÄ±äÌå£¬Í¬Ê±¿¼ÂÇ×î´óÖµºÍ×îÐ¡Öµ¡£

    Ô­Àí£º
        ¹éÒ»»¯£º
            ½«Ô¤²âÖµºÍÄ¿±êÖµ±ê×¼»¯Îª mean(x)£¬²¢¸ù¾Ý×î´óÖµºÍ×îÐ¡Öµµ÷Õû¡£
    Îó²î¼ÆËã£º
        ÀàËÆ norm_loss_with_min_max_normalization£¬Ê¹ÓÃ MSE ºÍÓàÏÒÏàËÆ¶Èµ÷Õû¡£
    '''
    if y_pred.size(0) > 1:  
        mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - mean_hat) / (eps + M_hat - m_hat)  # mean normalization
        y = (y - torch.mean(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_scaling(y_pred, y, alpha=[1, 1], p=2, detach=False):
    if y_pred.size(0) > 1:  
        normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p) 
        y_pred = y_pred / (eps + normalization)  # mean normalization
        y = y / (eps + torch.norm(y, p=p))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.