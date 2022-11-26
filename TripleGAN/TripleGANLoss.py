import torch
from torch import nn
from torch.nn import functional as F

from Auto_encoder import *
from Generator import *
from Classifier import *
from Discriminator import *
from scipy.stats import nakagami

class Loss_of_Discriminator(nn.Module):
    def __init__(self, device_num):
        super(Loss_of_Discriminator, self).__init__()
        self.device = torch.device("cuda:" + str(device_num))

    def forward(self, output_of_netD, bs_label, bs_unlabel):
        y_real = torch.ones((bs_label))  # 真
        y_fake_g = torch.zeros((bs_label))  # 假
        y_fake_c = torch.zeros((bs_unlabel)) # 假

        y_real = y_real.long()
        y_fake_g = y_fake_g.long()
        y_fake_c = y_fake_c.long()
        if torch.cuda.is_available():
            y_real = y_real.to(self.device)
            y_fake_g = y_fake_g.to(self.device)
            y_fake_c = y_fake_c.to(self.device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        sigmoid_output_of_netD = F.sigmoid(output_of_netD)

        loss_d_r = cross_entropy_loss(sigmoid_output_of_netD[0:bs_label],y_real)
        loss_d_f_g = cross_entropy_loss(sigmoid_output_of_netD[bs_label:bs_label*2],y_fake_g)
        loss_d_f_c = cross_entropy_loss(sigmoid_output_of_netD[bs_label*2:bs_label*2+bs_unlabel],y_fake_c)

        alpha = 0.5
        loss_d = loss_d_r + alpha * loss_d_f_g + (1 - alpha) * loss_d_f_c

        return loss_d

class Loss_of_Classifier(nn.Module):
    def __init__(self):
        super(Loss_of_Classifier, self).__init__()

    def forward(self,logits_all, target_all):
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        log_softmax_logits_all= F.log_softmax(logits_all)
        ce_l = cross_entropy_loss(log_softmax_logits_all[0:len(logits_all)//2],target_all[0:len(logits_all)//2])
        ce_g = cross_entropy_loss(log_softmax_logits_all[len(logits_all)//2:len(logits_all)],target_all[len(logits_all)//2:len(logits_all)])
        loss_c = ce_l + ce_g

        return loss_c

class Loss_of_Generator(nn.Module):
    def __init__(self, device_num):
        super(Loss_of_Generator, self).__init__()
        self.device = torch.device("cuda:" + str(device_num))

    def forward(self,d_logits_g):
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        y_real = torch.ones((len(d_logits_g)))  # 真
        y_real = y_real.long()
        if torch.cuda.is_available():
            y_real = y_real.to(self.device)
        sigmoid_d_logits_g = F.sigmoid(d_logits_g)
        loss_g = cross_entropy_loss(sigmoid_d_logits_g, y_real)

        return loss_g

class Loss_of_AutoEncoder(nn.Module):
    def __init__(self):
        super(Loss_of_AutoEncoder, self).__init__()

    def forward(self, output_of_NetA, input_of_NetA):
        loss_mse = torch.nn.MSELoss()
        loss_a = loss_mse(output_of_NetA, input_of_NetA)

        return loss_a

def z_obeys_nakagami(data_label, bs_data_label,m=1):
    '''
    data_label: 监督样本
    bs_data_label: batchsize of 监督样本
    m: Nakagami-m fading parameter
    '''
    z = torch.zeros((bs_data_label,4800))
    for i in range(bs_data_label):
        p = torch.mean(data_label[i,:])
        z[i,:] = torch.tensor(nakagami.rvs(m, loc = 0, scale = abs(p), size = 4800))

    return z


if __name__ == "__main__":
    auto_encoder = Auto_Encoder()
    generator = Encoder()
    classifier = Classifier()
    discriminator = Discriminator()

    #监督数据
    data_of_data_label = torch.randn((16,2,6000))
    label_of_data_label = torch.ones((16))
    label_of_data_label = label_of_data_label.long()

    #无监督数据
    data_of_data = torch.randn((16,2,6000))

    #自编码器输出数据，输入判别器的数据为out_of_netA[0],标签为label_of_data_label
    input_of_netA = torch.cat([data_of_data_label,data_of_data],dim=0)
    output_of_netA = auto_encoder(input_of_netA)

    #生成器输出数据，输入判别器的数据为g_fake_data,标签为label_of_data_label
    bs_label = 16
    z = z_obeys_nakagami(data_of_data_label, bs_label, m=1)
    data_of_gdata_label = generator(z,label_of_data_label)

    #分类器输出数据,输入判别器的数据为data_of_data,标签为label_ul
    bs_label = 16
    bs_unlabel = 16
    input_of_classifier = torch.cat([output_of_netA[0][0:bs_label,:], data_of_gdata_label, output_of_netA[0][bs_label:bs_label+bs_unlabel,:]],dim=0)
    output_of_netC = classifier(input_of_classifier)
    classifier_value_ul = F.log_softmax(output_of_netC[bs_label*2:bs_label*2+bs_unlabel])
    label_of_data_clabel = classifier_value_ul.argmax(dim=1, keepdim=True)
    label_of_data_clabel = label_of_data_clabel.squeeze(1)

    #判别器输出数据
    input_data_of_netD = torch.cat([output_of_netA[0][0:bs_label,:], data_of_gdata_label, output_of_netA[0][bs_label:bs_label+bs_unlabel,:]], dim=0)
    input_label_of_netD = torch.cat([label_of_data_label, label_of_data_label, label_of_data_clabel], dim=0)
    input_label_of_netD = input_label_of_netD.long()
    output_of_netD = discriminator(input_data_of_netD, input_label_of_netD)

    # 自编码器损失
    loss_netA = Loss_of_AutoEncoder(output_of_netA[1], input_of_netA)

    # 生成器损失
    bs_label = 16
    loss_netG = Loss_of_Generator(output_of_netD, bs_label)

    # 分类器损失
    loss_netC = Loss_of_Classifier(output_of_netC, label_of_data_label, bs_label)

    # 判别器损失
    bs_unlabel = 16
    loss_netD = Loss_of_Discriminator(output_of_netD, bs_label, bs_unlabel)

    optim_netA = torch.optim.Adam(auto_encoder.parameters(), lr=0.01)
    optim_netG = torch.optim.Adam(generator.parameters(), lr=0.01)
    optim_netC = torch.optim.Adam(classifier.parameters(), lr=0.01)
    optim_netD = torch.optim.Adam(discriminator.parameters(), lr=0.01)

    optim_netA.zero_grad()
    optim_netG.zero_grad()
    optim_netC.zero_grad()
    optim_netD.zero_grad()

    loss_netA.backward(retain_graph=True)
    loss_netG.backward(retain_graph=True)
    loss_netC.backward(retain_graph=True)
    loss_netD.backward()

    optim_netA.step()
    optim_netG.step()
    optim_netC.step()
    optim_netD.step()

    print("Loss_netA:{}, Loss_netG:{}, Loss_netC:{}, Loss_netD:{}".format(loss_netA,loss_netG,loss_netC,loss_netD))


