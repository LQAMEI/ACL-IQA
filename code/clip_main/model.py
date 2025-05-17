
import math
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from clip_main.adv_layer import ReverseLayerF
from IPython import embed

 
## hd, vd, cd, ad, rd convolutions
def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd','hd','vd', 'scd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':  
        return F.conv2d  
        
    if op_type == 'cd':   
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'   
            
            weights_c = weights.sum(dim=[2, 3]) - weights[:,:,1,1]
            weights_c = weights_c[:,:,None,None]
            
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
            return y - yc
        return func


    elif op_type == 'ad':
        
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            
            weights = weights.view(shape[0], shape[1], -1)
            weights_c = weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
            
            weights_c[:,:,4] = weights[:,:,4]*0
            weights_conv = (weights -weights_c).view(shape) 
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    elif op_type == 'rd':
        
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            
            buffer = torch.zeros(shape[0], shape[1], 5 * 5, dtype=weights.dtype).to(weights.device)
            weights = weights.view(shape[0], shape[1], -1)
            
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            
            buffer[:, :, 12] = weights[:, :, 0]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
                   
    elif op_type == 'hd':  
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 , 'kernel size for hd_conv should be 3'
            shape = weights.shape            
            weights_hd = torch.zeros(shape[0], shape[1], 3 * 3, dtype=weights.dtype).to(weights.device)
            weights = weights.view(shape[0], shape[1], -1)

            weights_hd[:, :, [0, 3, 6]] = weights[:, :, :]
            weights_hd[:, :, [2, 5, 8]] = -weights[:, :, :]
            weights_hd = weights_hd.view(shape[0], shape[1], 3, 3)
            y = F.conv2d(x, weights_hd, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    
    elif op_type == 'vd':  
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 , 'kernel size for vd_conv should be 3'
            shape = weights.shape
            weights_vd = torch.zeros(shape[0], shape[1], 3 * 3, dtype=weights.dtype).to(weights.device)
            weights = weights.view(shape[0], shape[1], -1)

            weights_vd[:, :, [0, 1, 2]] = weights[:, :, :]
            weights_vd[:, :, [6, 7, 8]] = -weights[:, :, :]
            weights_vd = weights_vd.view(shape[0], shape[1], 3, 3)
            y = F.conv2d(x, weights_vd, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
        
    elif op_type == 'scd':

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5, dtype=weights.dtype, device=weights.device)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5, dtype=weights.dtype)

            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:] * 2
            buffer[:, :, 12] = weights.sum(dim=[2])
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    
    else:
        print('impossible to be here unless you force that')
        return None

class Conv2d_Diff(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,op_type='cv'):
        super(Conv2d_Diff, self).__init__()
        assert op_type in ['cv', 'cd', 'ad', 'rd','hd','vd'], 'unknown op type: %s' % str(op_type)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if op_type in ['hd','vd']:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).half())  
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).half())
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).half())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.func = createConvFunc(op_type)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        out = self.func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class Conv2d_Adapter(nn.Module):
    def __init__(self, dim, adapter_dim, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, op_type='cv'):
        super(Conv2d_Adapter, self).__init__()
        
        self.adapter_down = nn.Linear(dim, adapter_dim)  
        self.adapter_up = nn.Linear(adapter_dim, dim)  
                
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.adapter_dim = adapter_dim
        
        self.adapter_conv = Conv2d_Diff(adapter_dim, adapter_dim, kernel_size, stride, padding, dilation, groups, bias,op_type)
        nn.init.zeros_(self.adapter_conv.weight)
        

    def forward(self, x):
        B, N, C = x.shape 
        
        x_down = self.adapter_down(x)  

        x_patch = x_down[:, 1:].reshape(B, 7, 7, self.adapter_dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)  
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 7 * 7, self.adapter_dim)  
        
        x_cls = x_down[:, :1].reshape(B, 1, 1, self.adapter_dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)  
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.adapter_dim)
        
        x_down = torch.cat([x_cls, x_patch], dim=1)
        x_up = self.adapter_up(x_down)  

        return x_up

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor. """
    def __init__(self, num_experts, gates):

        self._gates = gates 
        self._num_experts = num_experts

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]  
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert."""
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  
        return torch.split(inp_exp, self._part_sizes, dim=0)  

    def combine(self, expert_out, multiply_by_gates=False):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored."""

        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s."""
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Dual_Gated_MoElayer(nn.Module):
    """
    Call a sparsely dual-gating mixture of experts layer with convolutional experts, designed for adversarial and collaborative dual-path learning.
    """
    def __init__(self, dim=768, adapter_dim=8,adapter_type=['cv', 'cd','ad', 'rd','hd','vd'],noisy_gating=True, k=1): 
        super(Dual_Gated_MoElayer, self).__init__()
        self.noisy_gating = noisy_gating  
        self.num_experts = len(adapter_type)
        self.dim = dim  
        self.k = k   
        self.identity = nn.Identity()  

        adapter_experts = nn.ModuleList()
        for t in adapter_type:
            adapter_experts.append(Conv2d_Adapter(dim=dim,adapter_dim=adapter_dim,kernel_size=3,stride=1,padding=1,bias=True,op_type=t))

        self.num_experts = len(adapter_experts)
        self.adapter_experts = adapter_experts
        
        # gates for col and adv mechanism
        self.w_gate_col = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True) 
        self.w_gate_adv = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability."""
            
        eps = 1e-10  
        if x.shape[0] == 1:  
            return torch.tensor([0], device=x.device, dtype=x.dtype)

        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating."""
        batch = clean_values.size(0)   
        m = noisy_top_values.size(1)   
        
        top_values_flat = noisy_top_values.flatten()  
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        
        is_in = torch.gt(noisy_values, threshold_if_in)     
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, text_type, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538."""

        if text_type == 0:
            clean_logits = x @ self.w_gate_col
        else:
            clean_logits = x @ self.w_gate_adv
        
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)  

        zeros = torch.zeros_like(logits, requires_grad=True).to(top_k_gates.dtype) 
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, gate_x, text_type, x, loss_coef=1):  
        """Args:
        gate_x: The result of the interaction between image features and textual strategy features.
        ,tensor shape [batch_size, num_tokens, input_size] (144, 50, 768)
        text_type: ndicates which learning strategy to use: 0 for collaborative path, 1 for adversarial path.
        x: The image feature
        loss_coef: a scalar - multiplier on load-balancing losses
        """

        B, N, _ = x.shape  
        gate_x_global = torch.mean(gate_x,dim=1,keepdim=False)  
        gates, load = self.noisy_top_k_gating(gate_x_global,text_type, self.training) 
        
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)  
        gates = dispatcher.expert_to_gates() 
        expert_outputs = []
        
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0: continue  
            expert_output =  self.adapter_experts[i](expert_inputs[i])  
            expert_output = expert_output.reshape(expert_output.size(0), 50*self.dim)  
            expert_outputs.append(expert_output)
              
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, 50, self.dim)  
 
        return y,loss  

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        
        self._inplanes = width  
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, query, key, value, mask=None, return_attn=False):
        
        batch_size = query.size(0)
        
        q = self.q_proj(query) 
        k = self.k_proj(key)   
        v = self.v_proj(value)  
        
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)   

        context = torch.matmul(attn, v)    
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)  

        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output   


class MoEResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, text_d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter_topk=1):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        
        self.cross_attn = MultiHeadCrossAttention(d_model, d_model, text_d_model, n_head)

        self.ln_1 = LayerNorm(d_model)   

        self.mlp_col = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),   
            ("gelu", QuickGELU()),                      
            ("c_proj", nn.Linear(d_model * 4, d_model))    
        ]))
        
        self.mlp_adv = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),    
            ("gelu", QuickGELU()),                       
            ("c_proj", nn.Linear(d_model * 4, d_model)  )    
        ]))
        
        self.ln_2 = LayerNorm(d_model)
        
        self.attn_mask = attn_mask
        
        self.adapter_k = adapter_topk 
        if self.adapter_k>0:
            self.adapter_MoE = Dual_Gated_MoElayer(d_model,adapter_dim=8,k=self.adapter_k)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, text_type, past_key_values = None):
        if self.training == True:
            process_patch = 9
        else:
            process_patch = 16
        
        query = x.permute(1,0,2) 
        past_key_values = past_key_values.permute(1,0,2).repeat_interleave(process_patch, dim=0)    

        gates_x = self.cross_attn(query, past_key_values, past_key_values)  
        
        x = x + self.attention(self.ln_1(x))    
        
        hidden_states = self.ln_2(x)
        hidden_states = hidden_states.permute(1,0,2) 

        x_adapter, adapter_loss  = self.adapter_MoE(gates_x, text_type, hidden_states) 
        x_adapter = x_adapter.permute(1,0,2)  

        if text_type == 0:
            x = x + x_adapter + self.mlp_col(self.ln_2(x)) 
        else:
            x = x + x_adapter + self.mlp_adv(self.ln_2(x)) 

        return x, adapter_loss



class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        
        self.vision_width = vision_config[0]
        self.vision_layers =  vision_config[1]
        self.vision_heads = vision_config[2]
        self.topk = vision_config[3]
        
        self.text_width = text_config[0]
        self.text_layers = text_config[1]
        self.text_heads = text_config[2]
        self.attn_mask = text_config[3]
        
        assert self.vision_layers == self.text_layers
        
        vision_resblocks = []
        for i in range(self.vision_layers):
            if i%2==0:
                vision_resblocks.append(ResidualAttentionBlock(self.vision_width, self.vision_heads))
            else:
                vision_resblocks.append(MoEResidualAttentionBlock(self.vision_width,self.text_width, self.vision_heads, adapter_topk=self.topk))
        
        self.vision_resblocks = nn.Sequential(*vision_resblocks)
        
        self.text_resblocks = nn.Sequential(*[ResidualAttentionBlock(self.text_width, self.text_heads, attn_mask=self.attn_mask ) for _ in range(self.text_layers)])
        
         
    def forward(self, vision_embeds: torch.Tensor,  text_embeds: torch.Tensor, text_type):
        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds  

        past_key_values = None
        adapter_loss_list = []
        moe_loss = None
        
        for idx,blk in enumerate(self.vision_resblocks):
            
            past_key_values = text_hidden_states if hasattr(blk, 'adapter_k') else None
            
            if hasattr(blk, 'adapter_k'):
                vision_hidden_states, cur_adapter_loss = blk(
                        vision_hidden_states,
                        text_type = text_type,
                        past_key_values=past_key_values
                )
                adapter_loss_list.append(cur_adapter_loss)
            else:
                vision_hidden_states = blk(vision_hidden_states)

            text_layer_module = self.text_resblocks[idx]
            text_hidden_states = text_layer_module(text_hidden_states)
            
        if adapter_loss_list:
            moe_loss = torch.mean(torch.stack(adapter_loss_list))
        
        
        return vision_hidden_states, moe_loss
    

class UnimoModel(nn.Module):
    def __init__(self, 
                  embed_dim: int, 
                  
                  image_resolution: int, 
                  vision_patch_size: int, 
                  vision_width: int,   
                  vision_layers: Union[Tuple[int, int, int, int], int],  
                  
                  topk,
                  
                  context_length: int,
                  vocab_size: int,
                  transformer_width: int,  
                  transformer_heads: int,   
                  transformer_layers: int,  
                  
                  ):
        super().__init__()
        
        self.input_resolution = image_resolution
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=vision_width, kernel_size=vision_patch_size, stride=vision_patch_size, bias=False)

        scale = vision_width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(vision_width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_resolution // vision_patch_size) ** 2 + 1, vision_width))
        self.ln_pre = LayerNorm(vision_width)
        
        self.ln_post_col = LayerNorm(vision_width) 
        self.ln_post_adv = LayerNorm(vision_width)
        
        self.proj_col = nn.Parameter(scale * torch.randn(vision_width, embed_dim))
        self.proj_adv = nn.Parameter(scale * torch.randn(vision_width, embed_dim))
        
        self.topk = topk

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.text_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.attn_mask=self.build_attention_mask()
       
       
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))        
        
        self.transformer_layers = transformer_layers
        
        vision_heads = vision_width // 64  
        vision_config = [vision_width, vision_layers, vision_heads, topk]
        text_config = [transformer_width, transformer_layers, transformer_heads, self.attn_mask]
        
        self.encoder = UnimoEncoder(vision_config, text_config)
        
        self.theta = nn.Parameter(torch.tensor(0.5)) 
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.text_positional_embedding, std=0.01)


        proj_std = (self.encoder.text_width ** -0.5) * ((2 * self.encoder.text_layers) ** -0.5)
        attn_std = self.encoder.text_width ** -0.5
        fc_std = (2 * self.encoder.text_width) ** -0.5
        for block in self.encoder.text_resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.encoder.text_width ** -0.5)
            

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  
        return mask
        
    @property
    def dtype(self):
        return self.conv1.weight.dtype
      

    def encode(self, x: torch.Tensor, text: torch.Tensor, text_type):
        x = self.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  

        x_text = self.token_embedding(text).type(self.dtype)  
        x_text = x_text + self.text_positional_embedding.type(self.dtype)
        x_text = x_text.permute(1, 0, 2)  

        out, moe_loss = self.encoder(x, x_text, text_type)
        
        out = out.permute(1, 0, 2) 
        if text_type == 0:
            out = self.ln_post_col(out[:, 0, :])
            out = out @ self.proj_col
        else:
            out = self.ln_post_adv(out[:, 0, :])
            out = out @ self.proj_adv
        
        return out, moe_loss
        
        
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.text_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        
        text_hidden_states = x
        for idx in range(self.transformer_layers):   
            text_layer_module = self.encoder.text_resblocks[idx]
            text_hidden_states = text_layer_module(text_hidden_states)
            
        x = text_hidden_states
        
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
        
        
    def forward(self, image, text1, col_text, adv_text, text2=None, alpha=1):
        col, col_moe_loss = self.encode(image, col_text, 0) 
        adv, adv_moe_loss = self.encode(image, adv_text, 1) 
        
        moe_loss = col_moe_loss + adv_moe_loss
 
        col = col / col.norm(dim=1, keepdim=True)
        adv = adv / adv.norm(dim=1, keepdim=True)
        
        text1_features = self.encode_text(text1)    
        text1_features = text1_features / text1_features.norm(dim=1, keepdim=True)
        
        
        logit_scale = self.logit_scale.exp()
        logits_per_image_col = logit_scale * col @ text1_features.t()
        logits_per_image_adv = logit_scale * adv @ text1_features.t()
        logits_per_image = logits_per_image_col * self.theta + logits_per_image_adv*(1-self.theta)
 
        logits_per_image_col_2 = None
        logits_per_image_adv_2 = None

        if text2 is not None :
            text2_features = self.encode_text(text2)  
            text2_features = text2_features / text2_features.norm(dim=1, keepdim=True)   

            logits_per_image_col_2 = logit_scale * col @ text2_features.t()
            logits_per_image_adv_2 = logit_scale * adv @ text2_features.t()
            logits_per_image_adv_2 = ReverseLayerF.apply(logits_per_image_adv_2, alpha)

        return logits_per_image, logits_per_image_col_2, logits_per_image_adv_2, moe_loss


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if isinstance(attr, nn.Parameter):
                    attr.data = attr.data.half()
                elif attr is not None:
                    setattr(l, name, attr.half())
            '''        
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
            '''
    model.apply(_convert_weights_to_fp16)



def build_model(state_dict: dict, topk=1):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = UnimoModel(
        embed_dim,
        
        
        image_resolution, vision_patch_size, vision_width, vision_layers, topk,
        
        
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    file_path = r'state_dict.txt'
    with open(file_path, "w") as f:
        for key in state_dict.keys():
            f.write(key + "\n")
    
    
    new_state_dict = load_clip_weights(state_dict, vision_layers)
    model.load_state_dict(new_state_dict, strict = False)
    
    convert_weights(model)
    
    return model.eval()


def load_clip_weights(state_dict,vision_layers):
    new_state_dict = {}
    
    
    mapping = {
        'positional_embedding': 'text_positional_embedding',
        'visual.class_embedding': 'class_embedding',
        'visual.positional_embedding': 'positional_embedding',
        'visual.proj': 'proj_col',
        'visual.conv1.weight': 'conv1.weight',
        'visual.ln_pre.weight': 'ln_pre.weight',  
        'visual.ln_pre.bias': 'ln_pre.bias',      
        'visual.ln_post.weight': 'ln_post_col.weight',
        'visual.ln_post.bias': 'ln_post_col.bias',
        'text_projection': 'text_projection',
        'logit_scale': 'logit_scale',
        'token_embedding.weight': 'token_embedding.weight',
        'ln_final.weight': 'ln_final.weight',
        'ln_final.bias': 'ln_final.bias'
    }

    
    for old_key, new_key in mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]
    
    new_state_dict['ln_post_adv.weight'] = new_state_dict['ln_post_col.weight'] 
    new_state_dict['ln_post_adv.bias'] = new_state_dict['ln_post_col.bias'] 
    new_state_dict['proj_adv'] = new_state_dict['proj_col'] 

    
    for i in range(vision_layers):
        
        old_prefix = f'visual.transformer.resblocks.{i}'
        new_prefix = f'encoder.vision_resblocks.{i}'
        
        block_mapping = {
            'attn.in_proj_weight': 'attn.in_proj_weight',
            'attn.in_proj_bias': 'attn.in_proj_bias',
            'attn.out_proj.weight': 'attn.out_proj.weight',
            'attn.out_proj.bias': 'attn.out_proj.bias',
            'ln_1.weight': 'ln_1.weight',
            'ln_1.bias': 'ln_1.bias',
            'mlp.c_fc.weight': 'mlp.c_fc.weight',
            'mlp.c_fc.bias': 'mlp.c_fc.bias',
            'mlp.c_proj.weight': 'mlp.c_proj.weight',
            'mlp.c_proj.bias': 'mlp.c_proj.bias',
            'ln_2.weight': 'ln_2.weight',
            'ln_2.bias': 'ln_2.bias'
        }
        
        for old_suffix, new_suffix in block_mapping.items():
            old_key = f'{old_prefix}.{old_suffix}'
            new_key = f'{new_prefix}.{new_suffix}'
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]
        
        if i % 2 != 0:
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_col.c_fc.weight'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.weight'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_col.c_fc.bias'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.bias'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_col.c_proj.weight'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.weight'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_col.c_proj.bias'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.bias'] 
            
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_adv.c_fc.weight'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.weight'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_adv.c_fc.bias'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.bias'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_adv.c_proj.weight'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.weight'] 
            new_state_dict[f'encoder.vision_resblocks.{i}.mlp_adv.c_proj.bias'] = new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.bias']   
            
            del new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.weight']    
            del new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_fc.bias']  
            del new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.weight']  
            del new_state_dict[f'encoder.vision_resblocks.{i}.mlp.c_proj.bias'] 
            
            
        
        old_prefix = f'transformer.resblocks.{i}'
        new_prefix = f'encoder.text_resblocks.{i}'
        
        for old_suffix, new_suffix in block_mapping.items():
            old_key = f'{old_prefix}.{old_suffix}'
            new_key = f'{new_prefix}.{new_suffix}'
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]

        
    return new_state_dict


