# Code Implementation of the MambaIR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat



NEG_INF = -1000000


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

##local_scan代码
def local_scan(x, w=8, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, -1)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, -1)
    if flip:
        x = x.flip([-1])
    return x

def local_reverse(x, w=8, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, C, L]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, L = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if flip:
        x = x.flip([-1])
    if H % w != 0 or W % w != 0:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, Hg * w, Wg * w)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg * w, Wg * w)
        x = x[:, :, :H, :W].reshape(B, C, -1)
    else:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
    return x

def get_sample_img(x,h,w,level=1):
    ratio=2**level
    if(h%ratio!=0 or w%ratio!=0):
        newh,neww=math.ceil(h/ratio)*ratio,math.ceil(w/ratio)*ratio
        x=F.pad(x,(0,neww-w,0,newh-h))
    B,C,H,W=x.shape
    x=x.view(B,C,H//ratio,ratio,W//ratio,ratio).permute(0,3,5,1,2,4).contiguous()
    x=x.view(-1,C,H//ratio,W//ratio)
    return x

def reverse_sample_img(y,h,w,level=1):
    KB,C,H,W=y.shape
    ratio=2**level
    y=y.view(-1,ratio,ratio,C,H,W).permute(0,3,4,1,5,2).contiguous()
    y=y.view(-1,C,H*ratio,W*ratio)
    if(h%ratio!=0 or w%ratio!=0):
        y=y[:,:,:h,:w]
    return y




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            multi_scale=True,
            parallel=True,
            level_reverse=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.multi_scale = multi_scale
        self.parallel = parallel
        self.level_reverse = level_reverse
    

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        num_groups = 2 if multi_scale else 1
        param_groups = [
            self.init_group_params(
                d_inner=self.d_inner,
                d_state=self.d_state, 
                dt_rank=self.dt_rank,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                device=device,
                dtype=dtype
            ) for _ in range(num_groups)
        ]

        for i, (x_proj_w, dt_projs_w, dt_projs_b, a_logs, ds) in enumerate(param_groups):
            setattr(self, f'x_proj_weight_{i}', x_proj_w)
            setattr(self, f'dt_projs_weight_{i}', dt_projs_w) 
            setattr(self, f'dt_projs_bias_{i}', dt_projs_b)
            setattr(self, f'A_logs_{i}', a_logs)
            setattr(self, f'Ds_{i}', ds)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # self.fusion_conv = nn.Conv2d(self.d_inner * 3, self.d_inner, kernel_size=1)

    def init_group_params(
        self, 
        d_inner=None,      
        d_state=None,      
        dt_rank=None,      
        copies=4,          
        dt_scale=1.0,      
        dt_init="random",  
        dt_min=0.001,      
        dt_max=0.1,        
        dt_init_floor=1e-4,
        device=None,       
        dtype=None,        
        merge=True         
    ):
        """
        
        Returns:
            tuple: (x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds)
        """
        d_inner = d_inner if d_inner is not None else self.d_inner
        d_state = d_state if d_state is not None else self.d_state
        dt_rank = dt_rank if dt_rank is not None else self.dt_rank

        x_proj_weight = self.x_proj_init(
            d_inner, dt_rank, d_state,
            copies=copies, device=device, dtype=dtype
        )
        
        dt_projs_weight, dt_projs_bias = self.dt_projs_init(
            dt_rank, d_inner,
            dt_scale=dt_scale, dt_init=dt_init,
            dt_min=dt_min, dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            copies=copies, device=device, dtype=dtype
        )
        
        A_logs = self.A_log_init(
            d_state, d_inner, 
            copies=copies, device=device, merge=merge
        )
        
        Ds = self.D_init(
            d_inner, 
            copies=copies, device=device, merge=merge
        )
        
        return x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds

    @staticmethod
    def x_proj_init(d_inner, dt_rank, d_state, copies=1, device=None, dtype=None, merge=True):
        """Initialize x projection parameters
        Args:
            d_inner: inner dimension
            dt_rank: delta rank
            d_state: state dimension
            copies: number of copies (default 1)
            device: torch device
            dtype: torch dtype
            merge: whether to merge copies into one parameter
        Returns:
            nn.Parameter: x projection weights
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        x_projs = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(copies)
        ]
        x_proj_weight = torch.stack([t.weight for t in x_projs], dim=0)  # (copies, N, inner)
        return nn.Parameter(x_proj_weight)

    @staticmethod 
    def dt_projs_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, 
                      dt_max=0.1, dt_init_floor=1e-4, copies=1, device=None, dtype=None, merge=True):
        """Initialize delta projection parameters
        Args:
            dt_rank: delta rank
            d_inner: inner dimension
            dt_scale: scale factor for initialization
            dt_init: initialization type ("constant" or "random")
            dt_min: minimum delta value
            dt_max: maximum delta value 
            dt_init_floor: minimum floor value
            copies: number of copies
            device: torch device
            dtype: torch dtype
            merge: whether to merge copies
        Returns:
            tuple(nn.Parameter, nn.Parameter): weights and biases
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        dt_projs = [
            SS2D.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(copies)
        ]
        dt_projs_weight = torch.stack([t.weight for t in dt_projs], dim=0)  # (copies, inner, rank)
        dt_projs_bias = torch.stack([t.bias for t in dt_projs], dim=0)  # (copies, inner)
        
        return nn.Parameter(dt_projs_weight), nn.Parameter(dt_projs_bias)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @staticmethod
    def forward_core(x: torch.Tensor, x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds, d_state, dt_rank):
        """Static forward core computation
        Args:
            x: input tensor [B, C, H, W]
            x_proj_weight: x projection weights
            dt_projs_weight: delta projection weights
            dt_projs_bias: delta projection biases
            A_logs: A matrix logs
            Ds: D parameters
            d_state: state dimension
            dt_rank: delta rank
        Returns:
            tuple: (y1, y2, y3, y4) output tensors
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        # h v stacking
        x_hwwh = torch.stack([
            x.contiguous().view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        
        # h v hf vf concatenation
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B, 4, C, L)
        xs = xs.permute(0, 1, 3, 2) # (B, 4, L, C)

        # Local scan
        xs = torch.cat([
            local_scan(xs[:, i], H=H if i % 2 == 0 else W, W=W if i % 2 == 0 else H).unsqueeze(1)
            for i in range(K)
        ], dim=1)  # (B, K, C, new_L)

        new_L = xs.shape[-1]

        # Projections and splits
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, new_L), x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, new_L), dt_projs_weight)

        # Prepare inputs for selective scan
        xs = xs.float().view(B, -1, new_L)
        dts = dts.contiguous().float().view(B, -1, new_L) # (b, k * d, new_L)
        Bs = Bs.float().view(B, K, -1, new_L)
        Cs = Cs.float().view(B, K, -1, new_L) # (b, k, d_state, new_L)
        Ds = Ds.float().view(-1)
        As = -torch.exp(A_logs.float()).view(-1, d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1) # (k * d)

        # Selective scan
        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, new_L)

        # Local reverse
        out_y = torch.cat([
            local_reverse(out_y[:, i], H=H if i % 2 == 0 else W, W=W if i % 2 == 0 else H).unsqueeze(1)
            for i in range(K)
        ], dim=1)  # (B, K, C, L)

        # Final transformations
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        def process_group(x, scan_level):
            x_proj_weight = getattr(self, f'x_proj_weight_{scan_level}')
            dt_projs_weight = getattr(self, f'dt_projs_weight_{scan_level}')
            dt_projs_bias = getattr(self, f'dt_projs_bias_{scan_level}')
            A_logs = getattr(self, f'A_logs_{scan_level}')
            Ds = getattr(self, f'Ds_{scan_level}')
            
            y1, y2, y3, y4 = self.forward_core(
                x, x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds, self.d_state, self.dt_rank
            )
            
            assert y1.dtype == torch.float32
            return y1 + y2 + y3 + y4

        B, C, H, W = x.shape
        
        if not self.multi_scale:
            y = process_group(x, 0)
            y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        else:
            if not self.parallel:
                if not self.level_reverse:
                    y = process_group(x, 0)
                    
                    x_level1 = get_sample_img(y.view(B, C, H, W), H, W, level=1)
                    B1, C1, H1, W1 = x_level1.shape
                    y = process_group(x_level1, 1)
                    y = y.view(B1, C1, H1, W1)
                    y = reverse_sample_img(y, H, W, level=1)
                    
                    x_level2 = get_sample_img(y.view(B, C, H, W), H, W, level=2)
                    B2, C2, H2, W2 = x_level2.shape
                    y = process_group(x_level2, 2)
                    y = y.view(B2, C2, H2, W2)
                    y = reverse_sample_img(y, H, W, level=2)
                else:
                    x_level2 = get_sample_img(x, H, W, level=2)
                    B2, C2, H2, W2 = x_level2.shape
                    y = process_group(x_level2, 2)
                    y = y.view(B2, C2, H2, W2)
                    y = reverse_sample_img(y, H, W, level=2)
                    
                    x_level1 = get_sample_img(y.view(B, C, H, W), H, W, level=1)
                    B1, C1, H1, W1 = x_level1.shape
                    y = process_group(x_level1, 1)
                    y = y.view(B1, C1, H1, W1)
                    y = reverse_sample_img(y, H, W, level=1)
                    
                    y = process_group(y.view(B, C, H, W), 0)
                
                y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
            else:
                ##level0
                y0 = process_group(x, 0)
                y0 = y0.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # level 1
                x_level1 = get_sample_img(x, H, W, level=1)
                B1, C1, H1, W1 = x_level1.shape
                y1 = process_group(x_level1, 1)
                y1 = y1.view(B1, C1, H1, W1)
                y1 = reverse_sample_img(y1, H, W, level=1)
                y1 = y1.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # level 2
                # x_level2 = get_sample_img(x, H, W, level=2)
                # B2, C2, H2, W2 = x_level2.shape
                # y2 = process_group(x_level2, 2)
                # y2 = y2.view(B2, C2, H2, W2)
                # y2 = reverse_sample_img(y2, H, W, level=2)
                # y2 = y2.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # y = (y0 + y1 + y2) / 3
                y=(y0+y1)/2

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            multi_scale: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, 
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate, 
            multi_scale=multi_scale,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 multi_scale=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        if depth > 1:
            for i in range(depth-1):
                self.blocks.append(VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=self.mlp_ratio,
                    input_resolution=input_resolution,
                    is_light_sr=is_light_sr,
                    multi_scale=False
                ))
        self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[-1] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                multi_scale=multi_scale
            ))
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops




@ARCH_REGISTRY.register()
class DeflareMamba(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 mlp_ratio=2.,  ### expand
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(DeflareMamba, self).__init__()
        num_in_ch = 3  # 3+3
        num_out_ch = 6
        num_feat = 64
        self.img_range = img_range

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # ------------------------- 2, deep feature extraction ------------------------- #
        # self.num_layers = len(depths)
        self.num_enc_layers = 3
        self.num_dec_layers = 3
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[3]
        dec_dpr = enc_dpr[::-1]
        # refine_dpr = [drop_path_rate] * depths[7]


        self.enc0_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc0_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc0_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.enc0_layer = ResidualGroup(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[0],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=False  
        )
        self.enc0_norm = norm_layer(embed_dim)
        self.downsample_0 = nn.Conv2d(embed_dim, embed_dim*2, kernel_size=4, stride=2, padding=1)

        self.enc1_patch_embed = PatchEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 2,
            embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc1_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc1_patch_unembed = PatchUnEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 2,
            embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.enc1_layer = ResidualGroup(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[1],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=False  
        )
        self.enc1_norm = norm_layer(embed_dim * 2)
        self.downsample_1 = nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=4, stride=2, padding=1)

        self.enc2_patch_embed = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc2_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc2_patch_unembed = PatchUnEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.enc2_layer = ResidualGroup(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[2],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 4,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=False  
        )
        self.enc2_norm = norm_layer(embed_dim * 4)
        self.downsample_2 = nn.Conv2d(embed_dim*4, embed_dim*8, kernel_size=4, stride=2, padding=1)

        # self.enc3_patch_embed = PatchEmbed(
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     in_chans=embed_dim * 8,
        #     embed_dim=embed_dim * 8,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.enc3_patch_embed.patches_resolution
        # # self.patches_resolution[i_enc_layer] = patches_resolution
        # # return 2D feature map from 1D token sequence
        # self.enc3_patch_unembed = PatchUnEmbed(
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     in_chans=embed_dim * 8,
        #     embed_dim=embed_dim * 8,
        #     norm_layer=norm_layer if self.patch_norm else None)
        #
        # # build Residual State Space Group (RSSG)
        # self.enc3_layer = ResidualGroup(
        #     dim=embed_dim * 8,
        #     input_resolution=(patches_resolution[0], patches_resolution[1]),
        #     depth=depths[3],
        #     d_state=d_state,
        #     mlp_ratio=self.mlp_ratio,
        #     drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],  # no impact on SR results
        #     norm_layer=norm_layer,
        #     downsample=None,
        #     use_checkpoint=use_checkpoint,
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     resi_connection=resi_connection,
        #     is_light_sr=self.is_light_sr,
        #     multi_scale=False  # 最后一层设置为True
        # )
        # self.enc3_norm = norm_layer(embed_dim * 8)
        # self.downsample_3 = nn.Conv2d(embed_dim*8, embed_dim*16, kernel_size=4, stride=2, padding=1)

        self.bottle_patch_embed = PatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.bottle_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.bottle_patch_unembed = PatchUnEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.bottle_layer = ResidualGroup(
            dim=embed_dim * 8,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[3],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=conv_dpr,  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 8,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=True  # 最后一层设置为True
        )
        self.bottle_norm = norm_layer(embed_dim * 8)

        # Decoder
        # self.upsample_0 = nn.ConvTranspose2d(embed_dim * 16, embed_dim * 8, kernel_size=2, stride=2)
        # self.dec0_patch_embed = PatchEmbed(
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     in_chans=embed_dim * 16,
        #     embed_dim=embed_dim * 16,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.dec0_patch_embed.patches_resolution
        # # self.patches_resolution[i_enc_layer] = patches_resolution
        # # return 2D feature map from 1D token sequence
        # self.dec0_patch_unembed = PatchUnEmbed(
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     in_chans=embed_dim * 16,
        #     embed_dim=embed_dim * 16,
        #     norm_layer=norm_layer if self.patch_norm else None)
        #
        # # build Residual State Space Group (RSSG)
        # self.dec0_layer = ResidualGroup(
        #     dim=embed_dim * 16,
        #     input_resolution=(patches_resolution[0], patches_resolution[1]),
        #     depth=depths[5],
        #     d_state=d_state,
        #     mlp_ratio=self.mlp_ratio,
        #     drop_path=dec_dpr[:depths[5]],  # no impact on SR results
        #     norm_layer=norm_layer,
        #     downsample=None,
        #     use_checkpoint=use_checkpoint,
        #     img_size=img_size // 8,
        #     patch_size=patch_size,
        #     resi_connection=resi_connection,
        #     is_light_sr=self.is_light_sr,
        #     multi_scale=False  # 最后一层设置为True
        # )
        # self.dec0_norm = norm_layer(embed_dim * 16)

        self.upsample_1 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.dec1_patch_embed = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec1_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec1_patch_unembed = PatchUnEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.dec1_layer = ResidualGroup(
            dim=embed_dim * 8,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[4],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dec_dpr[:depths[4]],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 4,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=True 
        )
        self.dec1_norm = norm_layer(embed_dim * 8)

        self.upsample_2 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 2, kernel_size=2, stride=2)
        self.dec2_patch_embed = PatchEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec2_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec2_patch_unembed = PatchUnEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.dec2_layer = ResidualGroup(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[5],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 2,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=True 
        )
        self.dec2_norm = norm_layer(embed_dim * 4)

        self.upsample_3 = nn.ConvTranspose2d(embed_dim * 4, embed_dim, kernel_size=2, stride=2)
        self.dec3_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim*2,
            embed_dim=embed_dim*2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec3_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec3_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim*2,
            embed_dim=embed_dim*2,
            norm_layer=norm_layer if self.patch_norm else None)

        # build Residual State Space Group (RSSG)
        self.dec3_layer = ResidualGroup(
            dim=embed_dim*2,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[6],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            multi_scale=True  
        )
        self.dec3_norm = norm_layer(embed_dim*2)


#         self.refine_first_layer = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
#         self.refine_patch_embed = PatchEmbed(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=embed_dim,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         # num_patches = self.patch_embed.num_patches
#         patches_resolution = self.refine_patch_embed.patches_resolution
#         # self.patches_resolution[i_enc_layer] = patches_resolution
#         # return 2D feature map from 1D token sequence
#         self.refine_patch_unembed = PatchUnEmbed(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=embed_dim,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)

#         # build Residual State Space Group (RSSG)
#         self.refine_layer = ResidualGroup(
#             dim=embed_dim,
#             input_resolution=(patches_resolution[0], patches_resolution[1]),
#             depth=depths[7],
#             d_state=d_state,
#             mlp_ratio=self.mlp_ratio,
#             drop_path=refine_dpr,  # no impact on SR results
#             norm_layer=norm_layer,
#             downsample=None,
#             use_checkpoint=use_checkpoint,
#             img_size=img_size,
#             patch_size=patch_size,
#             resi_connection=resi_connection,
#             is_light_sr=self.is_light_sr,
#             multi_scale=True  # 最后一层设置为True
#         )
#         self.refine_norm = norm_layer(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == 'pixelshuffledirect' else False
        # stochastic depth

        # -------------------------3. high-quality image reconstruction ------------------------ #
        self.conv_last = nn.Conv2d(embed_dim*2, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        self.activation = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, patch_embed, patch_unembed, norm, layer):
        x_size = (x.shape[2], x.shape[3])
        x = patch_embed(x)  # N,L,C
        x = layer(x, x_size)
        x = norm(x)  # b seq_len c
        x = patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        y = self.conv_first(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.forward_features(y, self.enc0_patch_embed, self.enc0_patch_unembed, self.enc0_norm, self.enc0_layer)
        pool0 = self.downsample_0(conv0)
        conv1 = self.forward_features(pool0,self.enc1_patch_embed,self.enc1_patch_unembed,self.enc1_norm, self.enc1_layer)
        pool1 = self.downsample_1(conv1)
        conv2 = self.forward_features(pool1,self.enc2_patch_embed,self.enc2_patch_unembed,self.enc2_norm, self.enc2_layer)
        pool2 = self.downsample_2(conv2)
        # conv3 = self.forward_features(pool2,self.enc3_patch_embed,self.enc3_patch_unembed,self.enc3_norm, self.enc3_layer)
        # pool3 = self.downsample_3(conv3)

        #Bottleneck
        conv3 = self.forward_features(pool2,self.bottle_patch_embed,self.bottle_patch_unembed,self.bottle_norm, self.bottle_layer)

        #Decoder
        # up0 = self.upsample_0(conv4)
        # deconv0 = torch.cat((up0, conv3), dim=1)
        # deconv0 = self.forward_features(deconv0,self.dec0_patch_embed,self.dec0_patch_unembed,self.dec0_norm, self.dec0_layer)
        up1 = self.upsample_1(conv3)
        deconv1 = torch.cat((up1, conv2), dim=1)
        deconv1 = self.forward_features(deconv1,self.dec1_patch_embed,self.dec1_patch_unembed,self.dec1_norm, self.dec1_layer)
        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat((up2, conv1), dim=1)
        deconv2 = self.forward_features(deconv2,self.dec2_patch_embed,self.dec2_patch_unembed,self.dec2_norm, self.dec2_layer)
        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat((up3, conv0), dim=1)
        deconv3 = self.forward_features(deconv3,self.dec3_patch_embed,self.dec3_patch_unembed,self.dec3_norm, self.dec3_layer)

        # #Refine
        # y = self.refine_first_layer(deconv3) + y
        # y = self.forward_features(y,self.refine_patch_embed,self.refine_patch_unembed,self.refine_norm, self.refine_layer)
        y = self.conv_last(deconv3)

        # x = x / self.img_range + self.mean
        ##输出控制0，1
        y = self.activation(y)
        return y

    # def flops(self):
    #     flops = 0
    #     h, w = self.patches_resolution
    #     flops += h * w * 3 * self.embed_dim * 9
    #     flops += self.patch_embed.flops()
    #     for layer in self.layers:
    #         flops += layer.flops()
    #     flops += h * w * 3 * self.embed_dim * self.embed_dim
    #     flops += self.upsample.flops()
    #     return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr=False,
                 multi_scale=False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
            multi_scale=multi_scale
        )

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
