import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from SSRCNN_Complex import *
from get_dataset_10label import TrainDataset, TestDataset
from center_loss import CenterLoss

import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))

    return qlogq - qlogp

def addNoise(unlabelInput):
    num_channels = 2
    length = 4800
    d = np.random.randn(unlabelInput.shape[0], num_channels, length)
    Noise = [np.sqrt(np.sum(abs(unlabelInput[i].numpy())**2)/(50*np.sum(abs(d[i])**2)))*d[i] \
                     for i in range(unlabelInput.shape[0])]
    unlabelNoiseInput = unlabelInput + torch.Tensor(np.array(Noise))
    return unlabelNoiseInput

def train(model, loss_nll, loss_center, train_label_dataloader, train_unlabel_dataloader, optimizer_model, optimizer_cent, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    result_loss = 0
    nll_loss = 0
    nll_loss_ul = 0
    cent_loss = 0
    kl_loss = 0

    for (data_label, data_unlabel) in zip(train_label_dataloader, train_unlabel_dataloader):
        data, target = data_label
        data_ul, _ = data_unlabel
        data_ul_rand = addNoise(data_ul)
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            data_ul = data_ul.to(device)
            data_ul_rand = data_ul_rand.to(device)
            target = target.to(device)

        optimizer_model.zero_grad()
        optimizer_cent.zero_grad()

        data_all = torch.cat((data,data_ul,data_ul_rand),dim=0)
        output = model(data_all)

        output_0_l = output[0][0:len(data), :]
        output_0_ul = output[0][len(data):len(data) + len(data_ul), :]
        output_0_ul_rand = output[0][len(data) + len(data_ul):len(data) + len(data_ul) + len(data_ul_rand), :]

        output_1_l = output[1][0:len(data),:]
        output_1_ul = output[1][len(data):len(data)+len(data_ul),:]
        output_1_ul_rand =output[1][len(data)+len(data_ul):len(data)+len(data_ul)+len(data_ul_rand),:]

        classifier_value = F.log_softmax(output_1_l, dim=1)
        nll_loss_batch = loss_nll(classifier_value, target)

        classifier_value_ul = F.log_softmax(output_1_ul, dim=1)
        target_ul = classifier_value_ul.argmax(dim=1, keepdim=True)
        target_ul = target_ul.squeeze(1)
        nll_loss_batch_ul = torch.zeros(1).to(device)
        if epoch > 50:
            nll_loss_batch_ul = loss_nll(classifier_value_ul, target_ul)

        cent_loss_batch = loss_center(output_0_l, target)

        kl_loss_batch = kl_divergence_with_logit(output_1_ul, output_1_ul_rand)

        weight_cent = 0.003
        weight_kl = 1
        result_loss_batch = nll_loss_batch + weight_cent * cent_loss_batch + nll_loss_batch_ul + weight_kl * kl_loss_batch

        result_loss_batch.backward()
        optimizer_model.step()

        for param in loss_center.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_cent.step()

        nll_loss += nll_loss_batch.item()
        cent_loss += cent_loss_batch.item()
        nll_loss_ul += nll_loss_batch_ul.item()
        kl_loss += kl_loss_batch.item()
        result_loss += result_loss_batch.item()
        pred = classifier_value.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    nll_loss /= len(train_label_dataloader)
    cent_loss /= len(train_label_dataloader)
    nll_loss_ul /= len(train_unlabel_dataloader)
    kl_loss /= len(train_unlabel_dataloader)
    result_loss /= len(train_label_dataloader)

    print(
        'Train Epoch: {} \tClassifier_Loss(Label): {:.6f}, Classifier_Loss(Unlabel): {:.6f},  Center_Loss: {:.6f}, KL_Loss: {:.6f}, Combine_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            nll_loss,
            nll_loss_ul,
            cent_loss,
            kl_loss,
            result_loss,
            correct,
            len(train_label_dataloader.dataset),
            100.0 * correct / len(train_label_dataloader.dataset))
    )
    writer.add_scalar('Accuracy', 100.0 * correct / len(train_label_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/train', nll_loss, epoch)
    writer.add_scalar('Combined_Loss/train', result_loss, epoch)

def test(model, loss_model, test_dataloader, epoch, writer, device_num):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output[1], dim=1)
            test_loss += loss_model(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Loss/test', test_loss,epoch)

    return test_loss

def train_and_test(model, loss_model, loss_center, train_label_dataloader, train_unlabel_dataloader, val_dataloader, optimizer_model, optimizer_cent, epochs, writer, save_path, device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_model, loss_center, train_label_dataloader, train_unlabel_dataloader, optimizer_model, optimizer_cent, epoch, writer, device_num)
        test_loss = test(model, loss_model, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

def Data_prepared(n_classes, rand_num):
    X_train_label, X_train_unlabel, X_train, X_val, Y_train_label, Y_train_unlabel, Y_train, Y_val = TrainDataset(n_classes, rand_num)

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
    X_train_label, X_train_unlabel, X_train,  X_val, Y_train_label, Y_train_unlabel, Y_train, Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train_label = (X_train_label - min_value) / (max_value - min_value)
    X_train_unlabel = (X_train_unlabel - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train_label = X_train_label.transpose(0, 2, 1)
    X_train_unlabel= X_train_unlabel.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return  X_train_label, X_train_unlabel, X_val, Y_train_label, Y_train_unlabel, Y_val

class Config:
    def __init__(
        self,
        batch_size_label: int = 32,
        batch_size_unlabel: int = 288,
        test_batch_size: int = 32,
        epochs: int = 300,
        lr_model: float = 0.001,
        lr_cent: float = 0.05,
        log_interval: int = 10,
        n_classes: int = 10,
        save_path: str = 'model_weight/SSRCNN_n_classes_10_10label_90unlabel_rand30.pth',
        device_num: int = 0,
        rand_num: int = 30,
        ):
        self.batch_size_label = batch_size_label
        self.batch_size_unlabel = batch_size_unlabel
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_model = lr_model
        self.lr_cent = lr_cent
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num

def main():
    conf = Config()
    writer = SummaryWriter("logs_SSRCNN_n_classes_10_10label_rand30")
    device = torch.device("cuda:"+str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_label, X_train_unlabel, X_val, Y_train_label, Y_train_unlabel, Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_label_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    train_label_dataloader = DataLoader(train_label_dataset, batch_size=conf.batch_size_label, shuffle=True)

    train_unlabel_dataset = TensorDataset(torch.Tensor(X_train_unlabel), torch.Tensor(Y_train_unlabel))
    train_unlabel_dataloader = DataLoader(train_unlabel_dataset, batch_size=conf.batch_size_unlabel, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size_label, shuffle=True)

    model = SSRCNN()
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)

    loss_nnl = nn.NLLLoss()
    if torch.cuda.is_available():
        loss_nnl = loss_nnl.to(device)

    use_gpu = torch.cuda.is_available()
    loss_center = CenterLoss(num_classes = conf.n_classes, feat_dim = 128, use_gpu = use_gpu)

    optim_model = torch.optim.Adam(model.parameters(), lr=conf.lr_model)
    optim_centloss = torch.optim.Adam(loss_center.parameters(),lr= conf.lr_cent)

    train_and_test(model,
                   loss_model=loss_nnl,
                   loss_center = loss_center,
                   train_label_dataloader=train_label_dataloader,
                   train_unlabel_dataloader=train_unlabel_dataloader,
                   val_dataloader=val_dataloader,
                   optimizer_model=optim_model,
                   optimizer_cent=optim_centloss,
                   epochs=conf.epochs,
                   writer=writer,
                   save_path=conf.save_path,
                   device_num=conf.device_num)

if __name__ == '__main__':
   main()