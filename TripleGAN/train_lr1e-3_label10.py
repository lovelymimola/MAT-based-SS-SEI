import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Auto_encoder import *
from Generator import *
from Classifier import *
from Discriminator import *
from TripleGANLoss import *
from get_dataset_10label import *

import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(auto_encoder,
          generator,
          classifier,
          discriminator,
          loss_function_netA,
          loss_function_netG,
          loss_function_netC,
          loss_function_netD,
          label_dataloader,
          unlabel_dataloader,
          optim_netA,
          optim_netG,
          optim_netC,
          optim_netD,
          epoch,
          device_num,
          writer
          ):

    device = torch.device("cuda:" + str(device_num))
    loss_netA = 0
    loss_netG = 0
    loss_netC = 0
    loss_netD = 0
    correct = 0
    for (data_label, data_unlabel) in zip(label_dataloader, unlabel_dataloader):
        data_l, target = data_label
        data_ul, _ = data_unlabel
        target = target.long()
        z = z_obeys_nakagami(data_l, len(data_l), m=1)
        z2 = z_obeys_nakagami(data_l, len(data_l), m=1)
        if torch.cuda.is_available():
            data_l = data_l.to(device)
            data_ul = data_ul.to(device)
            z = z.to(device)
            z2 = z2.to(device)
            target = target.to(device)

        auto_encoder.train()
        generator.train()
        classifier.train()
        discriminator.train()

        optim_netA.zero_grad()
        optim_netG.zero_grad()
        optim_netC.zero_grad()
        optim_netD.zero_grad()

        '''
        自编码器
        输入：有标签数据data_l  无标签数据data_ul
        输出：数据的潜在特征features_l_ul = output_of_netA[0]
             数据的重构数据r_data_l_ul = output_of_netA[1]
             有标签数据的潜在特征features_l = output_of_netA[0][0:len(data_l),:] 
             无标签数据的潜在特征features_ul = output_of_netA[0][len(data_l):len(data_l)+len(data_ul),:]
             有标签数据的重构数据r_data_l = output_of_netA[1][0:len(data_l),:,:]
             无标签数据的重构数据r_data_ul = output_of_netA[1][len(data_l):len(data_l)+len(data_ul),:,:]
        '''
        data_l_ul = torch.cat([data_l, data_ul], dim=0)
        output_of_netA = auto_encoder(data_l_ul)
        r_data_l_ul = output_of_netA[1]
        ''''
        生成器
        输入：nakagami随机数 有标签数据的标签target
        输出：生成数据的潜在特征features_g
        '''
        features_g = generator(z, target)

        '''
        分类器
        输入：有标签数据的潜在特征features_l = output_of_netA[0][0:len(data_l),:] 
             生成数据的潜在特征features_g
             无标签数据的潜在特征features_ul = output_of_netA[0][len(data_l):len(data_l)+len(data_ul),:]
        输出：有标签数据的logits_l = output_of_netC[0:len(data_l),:]
             生成数据的logits_g = output_of_netC[len(data_l):len(data_l)*2,:]
             无标签数据的logits_ul = output_of_netC[len(data_l)*2:len(data_l)*2+len(data_ul),:]
        '''
        features_l = output_of_netA[0][0:len(data_l),:]
        features_ul = output_of_netA[0][len(data_l):len(data_l) + len(data_ul),:]
        input_of_classifier = torch.cat([features_l, features_g,features_ul], dim=0)
        output_of_netC = classifier(input_of_classifier)
        logits_l = output_of_netC[0:len(data_l), :]
        logits_g = output_of_netC[len(data_l):len(data_l) * 2, :]
        logits_ul = output_of_netC[len(data_l) * 2:len(data_l) * 2 + len(data_ul), :]

        classifier_value_ul = F.log_softmax(logits_ul)
        target_ul = classifier_value_ul.argmax(dim=1, keepdim=True)
        target_ul = target_ul.squeeze(1)
        '''
        判别器
        输入：有标签数据的潜在特征features_l = output_of_netA[0][0:len(data_l),:]
             生成数据的潜在特征features_g
             无标签数据的潜在特征features_ul = output_of_netA[0][len(data_l):len(data_l)+len(data_ul),:]
             
             有标签数据的标签target
             生成数据的标签target
             无标签数据的预测标签target_ul
             
        输出：判别结果d_logits
             有标签数据的判决结果d_logits_l = d_logits[0:len(data_l),:]
             生成数据的判决结果d_logits_g = d_logits[len(data_l):len(data_l)*2,:]
             无标签数据的判决结果d_logits_ul = d_logits[len(data_l)*2:len(data_l)*2+len(data_ul),:]
        '''
        input_data_of_netD = torch.cat([features_l, features_g,features_ul], dim=0)
        input_target_of_netD = torch.cat([target, target, target_ul], dim=0)
        input_label_of_netD = input_target_of_netD.long()
        d_logits = discriminator(input_data_of_netD, input_label_of_netD)
        d_logits_l = d_logits[0:len(data_l), :]
        d_logits_g = d_logits[len(data_l):len(data_l) * 2, :]
        d_logits_ul = d_logits[len(data_l) * 2:len(data_l) * 2 + len(data_ul), :]

        generator.eval()

        '''
        自编码器损失
        输入：重构数据r_data_l_ul, 真实数据data_l_ul
        输出：重构损失loss_netA
        '''
        loss_netA_batch = loss_function_netA(r_data_l_ul, data_l_ul)


        '''
        分类器损失
        输入：有标签数据的logits_l = output_of_netC[0:len(data_l),:]
             生成数据的logits_g = output_of_netC[len(data_l):len(data_l)*2,:]
             有标签数据的标签target
             生成数据的标签target
        输出：分类器损失loss_netC
        '''
        logits_all = torch.cat([logits_l, logits_g], dim=0)
        target_all = torch.cat([target, target], dim=0)
        loss_netC_batch = loss_function_netC(logits_all, target_all)

        '''
        判别器损失
        输入：判别结果d_logits
             有标签数据的batchsize
             无标签数据的batchsize
        输出：判决器损失loss_netD
        '''
        loss_netD_batch = loss_function_netD(d_logits, len(data_l), len(data_ul))

        loss_netA_batch.backward(retain_graph=True)
        loss_netC_batch.backward(retain_graph=True)
        loss_netD_batch.backward(retain_graph=True)

        optim_netA.step()
        optim_netC.step()
        optim_netD.step()

        auto_encoder.eval()
        classifier.eval()
        discriminator.eval()
        generator.train()
        '''
        生成器损失
        输入：生成数据的判决结果d_logits_g = d_logits[len(data_l):len(data_l)*2]
        输出：生成器损失loss_netG
        '''
        features_g = generator(z2, target)
        input_data_of_netD =  features_g
        input_target_of_netD = target
        input_label_of_netD = input_target_of_netD.long()
        d_logits = discriminator(input_data_of_netD, input_label_of_netD)
        d_logits_g = d_logits
        loss_netG_batch = loss_function_netG(d_logits_g)

        loss_netG_batch.backward()

        optim_netG.step()

        loss_netA += loss_netA_batch.item()
        loss_netG += loss_netG_batch.item()
        loss_netC += loss_netC_batch.item()
        loss_netD += loss_netD_batch.item()

        output = F.log_softmax(logits_l, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    loss_netA /= len(label_dataloader)
    loss_netG /= len(label_dataloader)
    loss_netC /= len(label_dataloader)
    loss_netD /= len(label_dataloader)

    print('Train Epoch: {} \tAuto_Encoder_Loss: {:.6f}, Generator_Loss, {:.6f}, Classifier_Loss: {:.6f}, Discriminator_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            loss_netA,
            loss_netG,
            loss_netC,
            loss_netD,
            correct,
            len(label_dataloader.dataset),
            100.0 * correct / len(label_dataloader.dataset))
    )

    writer.add_scalar('Accuracy/train', 100.0 * correct / len(label_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/train', loss_netC, epoch)

def validation(netA, netC, loss_function_netC, test_dataloader, epoch, device_num, writer):
    netA.eval()
    netC.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            '''
            自编码器
            输入：验证集数据data
            输出：验证集数据的潜在特征features = output_of_netA[0]
                 验证集数据的重构数据r_data = output_of_netA[1]
            '''
            output_of_netA = netA(data)
            features = output_of_netA[0]

            '''
            分类器
            输入：验证集数据的潜在特征features = output_of_netA[0]
            输出：验证集数据的logits = output_of_netC
            '''
            output_of_netC = netC(features)
            output = F.log_softmax(output_of_netC, dim=1)

            test_loss += loss_function_netC(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
        test_loss,
        correct,
        len(test_dataloader.dataset),
        100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/validation', test_loss, epoch)

    return test_loss

def train_and_validation(netA,
                         netG,
                         netC,
                         netD,
                         loss_function_netA,
                         loss_function_netG,
                         loss_function_netC,
                         loss_function_netD,
                         train_label_dataset,
                         train_unlabel_dataset,
                         val_dataset,
                         optim_netA,
                         optim_netG,
                         optim_netC,
                         optim_netD,
                         epochs,
                         netA_save_path,
                         netG_save_path,
                         netC_save_path,
                         netD_save_path,
                         device_num,
                         writer,
                         train_label_batch_size,
                         train_unlabel_batch_size):
    current_min_validation_loss = 100
    for epoch in range(1, epochs + 1):
        label_dataloader = DataLoader(train_label_dataset, batch_size=train_label_batch_size, shuffle=True)
        unlabel_dataloader = DataLoader(train_unlabel_dataset, batch_size=train_unlabel_batch_size,shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=train_label_batch_size, shuffle=True)
        train(netA,
              netG,
              netC,
              netD,
              loss_function_netA,
              loss_function_netG,
              loss_function_netC,
              loss_function_netD,
              label_dataloader,
              unlabel_dataloader,
              optim_netA,
              optim_netG,
              optim_netC,
              optim_netD,
              epoch,
              device_num,
              writer)

        validation_loss = validation(netA, netC, loss_function_netC, val_dataloader, epoch, device_num, writer)

        if validation_loss < current_min_validation_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_validation_loss, validation_loss))
            current_min_validation_loss = validation_loss
            torch.save(netA, netA_save_path)
            torch.save(netG, netG_save_path)
            torch.save(netC, netC_save_path)
            torch.save(netD, netD_save_path)

        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val = TrainDataset(n_classes, rand_num)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TrainDataset_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train_labeled = (X_train_labeled - min_value) / (max_value - min_value)
    X_train_unlabeled = (X_train_unlabeled - min_value) / (max_value - min_value)
    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train_labeled = X_train_labeled.transpose(0, 2, 1)
    X_train_unlabeled= X_train_unlabeled.transpose(0, 2, 1)
    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val

class Config:
    def __init__(
            self,
            train_label_batch_size: int = 32,
            train_unlabel_batch_size: int = 200,
            test_batch_size: int = 32,
            epochs: int = 300,
            lr_netA: float = 0.001,
            lr_netG: float = 0.001,
            lr_netC: float = 0.001,
            lr_netD: float = 0.001,
            n_classes: int = 10,
            netA_save_path: str = 'model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth',
            netG_save_path: str = 'model_weight/netG_n_classes_10_label10_unlabel90_rand30.pth',
            netC_save_path: str = 'model_weight/netC_n_classes_10_label10_unlabel90_rand30.pth',
            netD_save_path: str = 'model_weight/netD_n_classes_10_label10_unlabel90_rand30.pth',
            device_num: int = 0,
            rand_num: int = 30,
    ):
        self.train_label_batch_size = train_label_batch_size
        self.train_unlabel_batch_size = train_unlabel_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_netA = lr_netA
        self.lr_netG = lr_netG
        self.lr_netC = lr_netC
        self.lr_netD = lr_netD
        self.n_classes = n_classes
        self.netA_save_path = netA_save_path
        self.netG_save_path = netG_save_path
        self.netC_save_path = netC_save_path
        self.netD_save_path = netD_save_path
        self.device_num = device_num
        self.rand_num = rand_num

def main():
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs_TripleGAN_n_classes_10_10label_rand30")

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_label, X_train_unlabel, X_train, X_val, Y_train_label, Y_train_unlabel, Y_train, Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_label_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    train_unlabel_dataset = TensorDataset(torch.Tensor(X_train_unlabel), torch.Tensor(Y_train_unlabel))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    netA = Auto_Encoder()
    netG = Encoder()
    netC = Classifier()
    netD = Discriminator()
    if torch.cuda.is_available():
        netA = netA.to(device)
        netG = netG.to(device)
        netC = netC.to(device)
        netD = netD.to(device)

    loss_function_netA = Loss_of_AutoEncoder()
    loss_function_netG = Loss_of_Generator(conf.device_num)
    loss_function_netC = Loss_of_Classifier()
    loss_function_netD = Loss_of_Discriminator(conf.device_num)
    if torch.cuda.is_available():
        loss_function_netA = loss_function_netA.to(device)
        loss_function_netG = loss_function_netG.to(device)
        loss_function_netC = loss_function_netC.to(device)
        loss_function_netD = loss_function_netD.to(device)

    optim_netA = torch.optim.Adam(netA.parameters(), lr=conf.lr_netA)
    optim_netG = torch.optim.Adam(netG.parameters(), lr=conf.lr_netG)
    optim_netC = torch.optim.Adam(netC.parameters(), lr=conf.lr_netC)
    optim_netD = torch.optim.Adam(netD.parameters(), lr=conf.lr_netD)

    train_and_validation(netA,
                         netG,
                         netC,
                         netD,
                         loss_function_netA,
                         loss_function_netG,
                         loss_function_netC,
                         loss_function_netD,
                         train_label_dataset,
                         train_unlabel_dataset,
                         val_dataset,
                         optim_netA,
                         optim_netG,
                         optim_netC,
                         optim_netD,
                         conf.epochs,
                         conf.netA_save_path,
                         conf.netG_save_path,
                         conf.netC_save_path,
                         conf.netD_save_path,
                         conf.device_num,
                         writer,
                         conf.train_label_batch_size,
                         conf.train_unlabel_batch_size)

if __name__ == '__main__':
    main()




