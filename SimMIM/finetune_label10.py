import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn import *
from get_dataset_10label import *
import os #限制显卡
os.environ['CUDA_VISIBLE_DEVICES']='3' #限制显卡

def finetune(encoder,
             classifier,
             loss,
             train_dataloader_label,
             optimizer_encoder,
             optimizer_classifier,
             epoch,
             writer,
             device_num):
    encoder.train()
    classifier.train()
    device = torch.device("cuda:" + str(device_num))
    nll_loss = 0
    correct = 0
    for data_label in train_dataloader_label:
        data, target = data_label
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()
        feature = encoder(data)
        logit = classifier(feature)

        output = F.log_softmax(logit, dim=1)
        nll_loss_batch = loss(output, target)

        nll_loss_batch.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()

        nll_loss += nll_loss_batch.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    nll_loss /= len(train_dataloader_label)

    print('Train Epoch: {} \tClassifier Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        correct,
        len(train_dataloader_label.dataset),
        100.0 * correct / len(train_dataloader_label.dataset))
    )
    writer.add_scalar('Training Accuracy', 100.0 * correct / len(train_dataloader_label.dataset), epoch)
    writer.add_scalar('Training Loss', nll_loss, epoch)

def test(encoder, classifier, loss, test_dataloader, epoch, writer, device_num):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            feature = encoder(data)
            logit = classifier(feature)
            output = F.log_softmax(logit, dim=1)
            test_loss += loss(output, target).item()
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

    writer.add_scalar('Validation Accuracy', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Validation Loss', test_loss,epoch)

    return test_loss

def train_and_test(encoder,
                   classifier,
                   loss,
                   train_dataloader_label,
                   val_dataloader,
                   optimizer_encoder,
                   optimizer_classifier,
                   epochs,
                   writer,
                   encoder_save_path,
                   classifier_save_path,
                   device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        finetune(encoder, classifier, loss, train_dataloader_label, optimizer_encoder, optimizer_classifier, epoch, writer, device_num)
        test_loss = test(encoder, classifier, loss, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(encoder, encoder_save_path)
            torch.save(classifier, classifier_save_path)
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

def TestDataset_prepared(n_classes, rand_num):
    X_test, value_Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)
    X_test = X_test.transpose(0, 2, 1)

    return X_test, value_Y_test

def evaluate(encoder, classifier, test_dataloader, device_num):
    encoder.eval()
    classifier.eval()
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            feature = encoder(data)
            logit = classifier(feature)
            output = F.log_softmax(logit, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(correct / len(test_dataloader.dataset))

class Config:
    def __init__(
        self,
        batch_size_label: int = 32,
        batch_size_unlabel: int = 32,
        test_batch_size: int = 32,
        epochs: int = 300,
        lr: float = 0.001,
        mask_ratio: float = 0.5,
        encoder_original_save_path: str = 'model_weight/SimMIM_encoder_mask05_n_classes_10.pth',
        encoder_save_path: str = 'model_weight/SimMIM_encoder_mask05_n_classes_10_label10.pth',
        classifier_save_path: str = 'model_weight/SimMIM_classifier_mask05_n_classes_10_label10.pth',
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
        self.encoder_original_save_path = encoder_original_save_path
        self.encoder_save_path = encoder_save_path
        self.classifier_save_path = classifier_save_path
        self.device_num = device_num
        self.n_classes = n_classes
        self.rand_num = rand_num

def main():
    conf = Config()
    writer = SummaryWriter("logs_SimMIM_finetune_mask05_n_classes_10_label10")
    device = torch.device("cuda:" + str(conf.device_num))

    X_train_label, X_train_unlabel, X_train, X_val, value_Y_train_label, value_Y_train_unlabel, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_dataset_label = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(value_Y_train_label))
    train_dataloader_label = DataLoader(train_dataset_label, batch_size=conf.batch_size_label, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size_label, shuffle=True)

    encoder = torch.load(conf.encoder_original_save_path)
    classifier = Classifier()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=conf.lr)

    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)

    train_and_test(encoder,
                   classifier,
                   loss=loss,
                   train_dataloader_label=train_dataloader_label,
                   val_dataloader=val_dataloader,
                   optimizer_encoder=optim_encoder,
                   optimizer_classifier=optim_classifier,
                   epochs=conf.epochs,
                   writer=writer,
                   encoder_save_path=conf.encoder_save_path,
                   classifier_save_path=conf.classifier_save_path,
                   device_num=conf.device_num)

    encoder = torch.load(conf.encoder_save_path)
    classifier = torch.load(conf.classifier_save_path)
    X_test, value_Y_test = TestDataset_prepared(conf.n_classes, conf.rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(value_Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size_label, shuffle=True)
    evaluate(encoder, classifier, test_dataloader, conf.device_num)

if __name__ == '__main__':
   main()