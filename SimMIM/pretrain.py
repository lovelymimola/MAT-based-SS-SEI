import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn import *
from get_dataset_100label import *
import os #限制显卡
os.environ['CUDA_VISIBLE_DEVICES']='0' #限制显卡

def train(encoder,
          decoder,
          train_dataloader_label,
          train_dataloader_unlabel,
          mask_ratio,
          optimizer_encoder,
          optimizer_decoder,
          epoch,
          writer,
          device_num):
    encoder.train()
    decoder.train()
    device = torch.device("cuda:" + str(device_num))
    mse_loss = 0
    for (data_label, data_unlabel) in zip(train_dataloader_label,train_dataloader_unlabel):
        data, _ = data_label
        data_ul, _ = data_unlabel
        if torch.cuda.is_available():
            data = data.to(device)
            data_ul = data_ul.to(device)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        input = data
        bbx1, bbx2, input_masked = MaskData(input, mask_ratio)
        feature = encoder(input_masked)
        input_masked_r = decoder(feature)

        mask = torch.zeros((input.shape[0],input.shape[1],input.shape[2])).cuda()
        mask[:, :, bbx1: bbx2] = torch.ones((input.size()[1],bbx2-bbx1)).cuda()
        input = input.mul(mask)
        input_masked_r = input_masked_r.mul(mask)
        mse_loss_batch = F.mse_loss(input_masked_r, input, reduction='sum') / (bbx2-bbx1)
        mse_loss_batch.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        mse_loss += mse_loss_batch.item()


    mse_loss /= len(train_dataloader_label)

    print('Train Epoch: {} \tMSE Loss: {:.6f}, Number of Samples: {} \n'.format(
        epoch,
        mse_loss,
        len(train_dataloader_label.dataset)
    )
    )

    writer.add_scalar('Training Loss', mse_loss, epoch)

def test(encoder, decoder, test_dataloader, epoch, writer, device_num):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, _ in test_dataloader:
            if torch.cuda.is_available():
                data = data.to(device)
            z = encoder(data)
            data_r = decoder(z)
            test_loss += F.mse_loss(data_r, data, reduction='mean').item()

    test_loss /= len(test_dataloader)
    fmt = '\nValidation set: MSE loss: {:.6f}, Number of Samples: {}\n'
    print(
        fmt.format(
            test_loss,
            len(test_dataloader.dataset)
        )
    )

    writer.add_scalar('Validation Loss', test_loss,epoch)

    return test_loss

def train_and_test(encoder,
                   decoder,
                   train_dataloader_label,
                   train_dataloader_unlabel,
                   mask_ratio,
                   val_dataloader,
                   optimizer_encoder,
                   optimizer_decoder,
                   epochs,
                   writer,
                   encoder_save_path,
                   decoder_save_path,
                   device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(encoder, decoder, train_dataloader_label, train_dataloader_unlabel, mask_ratio, optimizer_encoder, optimizer_decoder, epoch, writer, device_num)
        test_loss = test(encoder, decoder, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(encoder, encoder_save_path)
            torch.save(decoder, decoder_save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

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
    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_train_labeled = (X_train_labeled - min_value) / (max_value - min_value)
    X_train_unlabeled = (X_train_unlabeled - min_value) / (max_value - min_value)
    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train_labeled = X_train_labeled.transpose(0, 2, 1)
    X_train_unlabeled = X_train_unlabeled.transpose(0, 2, 1)
    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val

class Config:
    def __init__(
        self,
        batch_size_label: int = 32,
        batch_size_unlabel: int = 32,
        test_batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        mask_ratio: float = 0.5,
        encoder_save_path: str = 'model_weight/SimMIM_encoder_mask05_n_classes_10.pth',
        decoder_save_path: str = 'model_weight/SimMIM_decoder_mask05_n_classes_10.pth',
        device_num: int = 0,
        n_classes: int = 10,
        rand_num: int = 30,
        ):
        self.batch_size_label = batch_size_label
        self.batch_size_unlabel = batch_size_unlabel
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.device_num = device_num
        self.n_classes = n_classes
        self.rand_num = rand_num

def main():
    conf = Config()
    writer = SummaryWriter("logs_SimMIM_mask05_n_classes_10")
    device = torch.device("cuda:" + str(conf.device_num))

    X_train_label, X_train_unlabel, X_train, X_val, value_Y_train_label, value_Y_train_unlabel, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_dataset_label = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(value_Y_train_label))
    train_dataloader_label = DataLoader(train_dataset_label, batch_size=conf.batch_size_label, shuffle=True)

    train_dataset_unlabel = TensorDataset(torch.Tensor(X_train_unlabel), torch.Tensor(value_Y_train_unlabel))
    train_dataloader_unlabel = DataLoader(train_dataset_unlabel, batch_size=conf.batch_size_unlabel, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size_label, shuffle=True)

    encoder = Encoder()
    decoder = Decoder()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=conf.lr)

    train_and_test(encoder,
                   decoder,
                   train_dataloader_label=train_dataloader_label,
                   train_dataloader_unlabel=train_dataloader_unlabel,
                   mask_ratio = conf.mask_ratio,
                   val_dataloader=val_dataloader,
                   optimizer_encoder=optim_encoder,
                   optimizer_decoder=optim_decoder,
                   epochs=conf.epochs,
                   writer=writer,
                   encoder_save_path=conf.encoder_save_path,
                   decoder_save_path=conf.decoder_save_path,
                   device_num=conf.device_num)

if __name__ == '__main__':
   main()