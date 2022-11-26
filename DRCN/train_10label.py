import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from get_dataset_10label import TrainDataset, TestDataset
from DRCN_Complex import *

import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(encoder, decoder, classifier, loss_nll, loss_mse, train_label_dataloader, train_dataloader, optimizer_encoder, optimizer_decoder, optimizer_classifier, epoch, writer, device_num):
    encoder.train()
    decoder.train()
    classifier.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    result_loss = 0
    nll_loss = 0
    mse_loss = 0
    for (data_label, data_all) in zip(train_label_dataloader, train_dataloader):
        data, target = data_label
        data_re, _ = data_all

        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            #data = data.unsqueeze(-1)
            target = target.to(device)

            data_re = data_re.to(device)
            #data_re = data_re.unsqueeze(-1)

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        optimizer_classifier.zero_grad()

        features = encoder(data)
        output = F.log_softmax(classifier(features), dim=1)
        nll_loss_batch = loss_nll(output, target)

        features_re = encoder(data_re)
        re_input = decoder(features_re)
        mse_loss_batch = loss_mse(re_input, data_re)

        lamda_CAE = 0.75
        result_loss_batch = (1-lamda_CAE) * nll_loss_batch + lamda_CAE * mse_loss_batch

        result_loss_batch.backward(retain_graph=True)
        nll_loss_batch.backward(retain_graph=True)
        mse_loss_batch.backward()

        optimizer_encoder.step()
        optimizer_classifier.step()
        optimizer_decoder.step()

        nll_loss += nll_loss_batch.item()
        mse_loss += mse_loss_batch.item()
        result_loss += result_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    nll_loss /= len(train_label_dataloader)
    mse_loss /= len(train_dataloader)
    result_loss = nll_loss + lamda_CAE * mse_loss

    print('Train Epoch: {} \tEncoder_Loss: {:.6f}, Decoder_Loss, {: 6f}, Classifier_Loss: {: 6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        result_loss,
        mse_loss,
        nll_loss,
        correct,
        len(train_label_dataloader.dataset),
        100.0 * correct / len(train_label_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_label_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/train', nll_loss, epoch)
    writer.add_scalar('MSE_Loss/train', mse_loss, epoch)


def test(encoder, classifier, loss_nll, test_dataloader, epoch, writer, device_num):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #data = data.unsqueeze(-1)
                target = target.to(device)

            features = encoder(data)
            output = F.log_softmax(classifier(features), dim=1)

            test_loss += loss_nll(output, target).item()
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

    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Loss/validation', test_loss,epoch)

    return test_loss

def train_and_test(encoder,
                   decoder,
                   classifier,
                   loss_nll,
                   loss_mse,
                   train_label_dataloader,
                   train_dataloader,
                   val_dataloader,
                   optimizer_encoder,
                   optimizer_decoder,
                   optimizer_classifier,
                   epochs,
                   writer,
                   encoder_save_path,
                   classifier_save_path,
                   device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(encoder, decoder, classifier, loss_nll, loss_mse, train_label_dataloader, train_dataloader, optimizer_encoder, optimizer_decoder, optimizer_classifier, epoch, writer, device_num)
        test_loss = test(encoder, classifier, loss_nll, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new encoder and classifier weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(encoder, encoder_save_path)
            torch.save(classifier, classifier_save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        for name, param in classifier.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train, X_val, Y_train_labeled, Y_train, Y_val = TrainDataset(n_classes, rand_num)

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
    X_train_labeled, X_train, X_val, Y_train_labeled, Y_train, Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train_labeled = (X_train_labeled - min_value) / (max_value - min_value)
    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train_labeled = X_train_labeled.transpose(0, 2, 1)
    X_train= X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return  X_train_labeled, X_train, X_val, Y_train_labeled, Y_train, Y_val

class Config:
    def __init__(
        self,
        train_label_batch_size: int = 32,
        train_batch_size: int = 320,
        test_batch_size: int = 32,
        epochs: int = 300,
        lr_encoder: float = 0.001,
        lr_decoder: float = 0.001,
        lr_classifier: float = 0.001,
        log_interval: int = 10,
        n_classes: int = 10,
        encoder_save_path: str = 'model_weight/encoder_complex_n_classes_10_label10_unlabel90_rand30.pth',
        classifier_save_path: str = 'model_weight/classifier_complex_n_classes_10_label10_unlabel90_rand30.pth',
        device_num: int = 0,
        rand_num: int = 30,
        ):
        self.train_label_batch_size = train_label_batch_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_classifier = lr_classifier
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.classifier_save_path = classifier_save_path
        self.device_num = device_num
        self.rand_num = rand_num

def main():
    conf = Config()
    writer = SummaryWriter("logs_DRCN_10label_rand30")
    device = torch.device("cuda:"+str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_label, X_train, X_val, Y_train_label, Y_train, Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_label_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    train_label_dataloader = DataLoader(train_label_dataset, batch_size=conf.train_label_batch_size, shuffle=True)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.train_label_batch_size, shuffle=True)

    encoder = Encoder()
    decoder = Decoder()
    classifier = Classifier()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        classifier = classifier.to(device)

    loss_nll = nn.NLLLoss()
    loss_mse = nn.MSELoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)
        loss_mse = loss_mse.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    optim_decoder = torch.optim.Adam(decoder.parameters(),lr=conf.lr_decoder)
    optim_classifier = torch.optim.Adam(classifier.parameters(),lr=conf.lr_classifier)

    train_and_test(encoder=encoder,
                   decoder=decoder,
                   classifier=classifier,
                   loss_nll=loss_nll,
                   loss_mse = loss_mse,
                   train_label_dataloader=train_label_dataloader,
                   train_dataloader = train_dataloader,
                   val_dataloader=val_dataloader,
                   optimizer_encoder=optim_encoder,
                   optimizer_decoder=optim_decoder,
                   optimizer_classifier=optim_classifier,
                   epochs=conf.epochs,
                   writer=writer,
                   encoder_save_path=conf.encoder_save_path,
                   classifier_save_path= conf.classifier_save_path,
                   device_num=conf.device_num)

if __name__ == '__main__':
   main()