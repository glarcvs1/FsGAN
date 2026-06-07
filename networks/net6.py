import torch
import torchvision.transforms.functional
from torch import nn
import numpy as np
from torch import autograd
from networks.blocks import *
import random
from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
from networks.ganblock import *
#修改resnet网络

class DaGAN6(nn.Module):
    def __init__(self, config):
        super(DaGAN6, self).__init__()
        self.gen = Gen(config)
        self.dis = Discriminator(config['dis'])
        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']

    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, similarity, indices_feat, indices_ref, k,s = self.gen(xs)

            loss_recon=[]
            for i in range(k):
                loss_recon1 = local_recon_criterion(xs, fake_x, similarity, indices_feat[i], indices_ref[i], i, s=s)  #局部重建损失
                loss_recon1 = loss_recon1 * self.w_recon / k
                loss_recon.append(loss_recon1)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())
            loss_recon_sum=loss_recon[0]
            for i in range(len(loss_recon)-1):
                loss_recon_sum += loss_recon[i+1]

            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_recon_sum + loss_adv_gen + loss_cls_gen
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1)
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

            loss_reg_dis = loss_reg_dis * self.w_gp
            loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg

class Gen(nn.Module):
    def __init__(self,config):
        super(Gen, self).__init__()
        self.encoder = unet_encoder(32)
        self.decoder = unet_decoder(32,config)
        self.fusion = LocalFusionModule(inplanes=128, rate=config['gen']['rate'])

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(b*k, C, H, W)
        querys, encoder_layers = self.encoder(xs)

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        feat_gen=[]
        indices_feat=[]
        indices_ref=[]
        s= querys.size()[-2]
        for i in range(k):
            c, h, w = querys.size()[-3:]
            querys = querys.view(b, k, c, h, w)
            base_feat = querys[:, i, :, :, :]
            feat_gen1, indices_feat1, indices_ref1 = self.fusion(base_feat, querys, i, similarity)
            feat_gen.append(feat_gen1)
            indices_feat.append(indices_feat1)
            indices_ref.append(indices_ref1)
        feat_gens=torch.stack(feat_gen,dim=1)


        fake_x = self.decoder(feat_gens,encoder_layers, similarity,b,k)

        return fake_x, similarity, indices_feat, indices_ref,k,s


class unet_encoder(nn.Module):
    def __init__(self,ch):
        super(unet_encoder, self).__init__()
        self.conv1 = Conv2dBlock(3, ch, 5, 1, 2,
                                 norm='none',
                                 activation='tanh',
                                 pad_type='reflect')
        # encode1=[ActFirstResBlock(ch,ch*2)]
        encode1 = [Conv2dBlock(ch, ch * 2, 3, 2, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        self.encode1 = nn.Sequential(*encode1)
        # encode2 = [ActFirstResBlock(ch*2, ch*2)]
        encode2 = [Conv2dBlock(ch * 2, ch * 2, 3, 2, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        self.encode2 = nn.Sequential(*encode2)
        # encode3 = [ActFirstResBlock(ch*2, ch*2)]
        encode3 = [Conv2dBlock(ch * 2, ch * 2, 3, 2, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        self.encode3 = nn.Sequential(*encode3)
        encode4 = [Conv2dBlock(ch * 2, ch * 4, 3, 1, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        # encode4 += [nn.AvgPool2d(kernel_size=2, stride=2)]
        self.encode4 = nn.Sequential(*encode4)

    def forward(self,x):
        encoder_layers = []
        x=self.conv1(x)
        lay1=self.encode1(x)
        encoder_layers.append(lay1)
        lay2 = self.encode2(lay1)
        encoder_layers.append(lay2)
        lay3 = self.encode3(lay2)
        encoder_layers.append(lay3)
        lay4 = self.encode4(lay3)
        encoder_layers.append(lay4)
        return lay4,encoder_layers

class unet_decoder(nn.Module):
    def __init__(self,ch,k):
        super(unet_decoder, self).__init__()
        self.concat = CropAndConcat()
        self.conv2 = nn.Conv2d(k['n_sample_train'] * ch * 4, ch * 2, (1, 1))
        decode1 = [Conv2dBlock(ch * 2, ch * 2, 3, 1, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        decode1 += [UpSample(ch * 2, ch * 2)]
        self.decode1 = nn.Sequential(*decode1)
        self.att1 = attention(ch * 2, ch * 2)
        decode2 = [Conv2dBlock(ch * 4, ch * 2, 3, 1, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        decode2 += [UpSample(ch * 2, ch * 2)]
        self.decode2 = nn.Sequential(*decode2)
        self.att2 = attention(ch * 2, ch * 2)
        decode3 = [Conv2dBlock(ch * 4, ch * 2, 3, 1, 1,
                               norm='bn',
                               activation='lrelu',
                               pad_type='reflect')]
        decode3 += [UpSample(ch * 2, ch * 2)]
        self.decode3 = nn.Sequential(*decode3)
        self.conv1 = Conv2dBlock(ch * 2, 3, 5, 1, 2,
                                 norm='none',
                                 activation='tanh',
                                 pad_type='reflect')
        self.classify = AttentionalClassify()
    def forward(self,x,encoder_layers,similarity,b,k):
        b1,k1,c1,w1,h1=x.size()
        x=x.view(b,k*c1,w1,h1)
        x=self.conv2(x)
        output=self.decode1(x)
        sim=similarity.view(similarity.size()[0],similarity.size()[1],1,1,1)
        b1, c1, w1, h1 = encoder_layers[-3].size()
        encoder_layerss = encoder_layers[-3].view(b, k, c1,w1,h1)
        current_encoder=torch.multiply(sim,encoder_layerss)
        encoder_s_attention=self.att1(current_encoder,output)
        encoder_s_attention = torch.stack(encoder_s_attention, dim=1)
        encoder_s_attention = torch.sum(encoder_s_attention, dim=1)
        output= self.concat(output,encoder_s_attention)
        output=self.decode2(output)
        b1, c1, w1, h1 = encoder_layers[-4].size()
        encoder_layerss = encoder_layers[-4].view(b, k, c1, w1, h1)
        current_encoder = torch.multiply(sim, encoder_layerss)
        encoder_s_attention = self.att2(current_encoder, output)
        encoder_s_attention = torch.stack(encoder_s_attention, dim=1)
        encoder_s_attention = torch.sum(encoder_s_attention, dim=1)
        output=self.concat(output,encoder_s_attention)
        output=self.decode3(output)
        #output = torch.cat([output,encoder_layers[-4]],dim=0)
        output=self.conv1 (output)
        return output

class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index] #基本图片相似度
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1) #其他图片相似度

        # take ref:(32, 2, 128,  8,8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)   #输入的四维特征进行转化为三维
        w_feat = w_feat.permute(0, 2, 1).contiguous()  #将通道信息放最后
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)在指定维度上的输入规范化。

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1) #再将其转为
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)第一维的归一化

        # local selection 局部选择
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num 随机挑选num个range（h*w）的数据，组成一个矩阵

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*num)通道的信息

        # local matching局部匹配
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*num*128)通道的信息复制
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)   #每一张候选图
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64) 选择某一张w_ref
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1) # 返回 fx中第三个维度取最大的一个值
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)squeeze(-1)：去除最后维度值为1的维度；squeeze(0)：代表若第一维度值为1则去除第一维度
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12) #挑选每个ref的特征
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement 局部置换
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']   #
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


if __name__ == '__main__':
    config = {}
    b=32
    k=3
    x = torch.randn(b, k,3, 128, 128).cuda()
    gen=Gen().cuda()
    fake_x, similarity, indices_feat, indices_ref,k=gen(x)
    print(fake_x.size())

