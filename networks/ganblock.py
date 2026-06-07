import torch
import torchvision.transforms.functional
from torch import nn
from networks.blocks import *

class CropAndConcat(nn.Module):
    """
    ### 裁剪和连接特征图

    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """
        # Crop the feature map from the contracting path to the size of the current feature map
        #contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x
def flatten(x):
    x=x.view(x.size()[0],x.size()[1],-1)
    return x

class attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(attention, self).__init__()
        self.convf=Conv2dBlock(in_dim, out_dim//8,(1,1),1)
        self.convg = Conv2dBlock(in_dim, out_dim//8, (1,1), 1)
        self.convh = Conv2dBlock(in_dim, out_dim, (1,1), 1)
        self.convv=Conv2dBlock(in_dim, out_dim, (1,1), 1)
        self.gamma=nn.Parameter(torch.zeros(1))
    def forward(self, ff,bf):
        encoder_s_attention=[]
        for i in range(ff.size()[1]):
            ff1=ff[:,i,:,:,:]
            f=self.convf(ff1)  #[b,c',h,w]
            g=self.convg(bf)
            h=self.convh(ff1)
            s=torch.matmul(torch.transpose(flatten(g), 1, 2), flatten(f)) #[b,N,N]  N=h*w
            beta= torch.softmax(s,dim=-1)
            o=torch.matmul(flatten(h),beta) #[b,c,N]
            o=o.view(ff1.size()[0],ff1.size()[1],ff1.size()[2],ff1.size()[3])
            o=self.convv(o)
            x=self.gamma*o+ff1
            encoder_s_attention.append(x)
        return encoder_s_attention



class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, use_cbam=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding 初始化填充
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization 初始化规范化
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation 初始化激活
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

        if use_cbam:
            self.cbam = CBAM(out_dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_cbam=False):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)
        if use_cbam:
            self.cbam = CBAM(dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.cbam:
            out = self.cbam(x)
        out += residual
        return out

class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = max(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out

class AttentionalClassify(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, similarities, x):
        b, k= similarities.size()
        similarities = similarities.view(b, 1, k)
        b, k, z = x.size()
        preds=torch.matmul(similarities,x)
        preds=preds.view(b, z)
        return preds

class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)
if __name__ == '__main__':
    b=32
    k=3
    f=[]
    x = torch.randn(b, k,1,1,1).cuda()
    g = torch.randn(b, k, 3, 128, 128).cuda()
    for i in range(5):
        y = torch.randn(b, k,128).cuda()
        f.append(y)
    #x.view(1,2,-1)
    #y=x[:, -1, :]
    print(y.size())
    k=torch.stack(f,dim=1)
    k=torch.sum(k,dim=1)
    k=torch.softmax(k,dim=-1)
    print(k.size())