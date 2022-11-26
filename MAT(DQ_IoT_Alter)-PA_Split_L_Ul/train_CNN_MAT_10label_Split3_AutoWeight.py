import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn import *
from get_dataset_10label import TrainDataset, TestDataset
from ProxyAnchorLoss import Proxy_Anchor
from VAT2 import VAT
from AutomaticWeightedLoss import AutomaticWeightedLoss
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

def train(model, loss, loss_ProxyAnchor, train_dataloader, VAT_dataloader, optimizer1, awl1, optimizer2, awl2, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    classifier_loss_label = 0
    classifier_loss_unlabel = 0
    proxy_anchor_loss_label = 0
    proxy_anchor_loss_unlabel = 0
    result_loss = 0
    vat_loss_label = 0
    vat_loss_unlabel = 0
    vat_loss_function = VAT(model)
    for (data_nnl, data_vat) in zip(train_dataloader,VAT_dataloader):
        data, target = data_nnl
        data_vat, target_vat = data_vat
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
            data_vat = data_vat.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        if epoch < 100:
            output = model(data)
            feature_label = output[0]
            logits_label = output[1]
            classifier_loss_label_batch = F.cross_entropy(logits_label, target, reduction='mean')
            classifier_loss_unlabel_batch = 0.0 * classifier_loss_label_batch
        if epoch >= 100:
            output = model(torch.cat([data, data_vat]))
            feature = output[0]
            logits = output[1]
            feature_label = feature[:len(data)]
            feature_unlabel = feature[len(data):len(data) + len(data_vat)]
            logits_label = logits[:len(data)]
            logits_unlabel = logits[len(data):len(data) + len(data_vat)]
            classifier_loss_label_batch = F.cross_entropy(logits_label, target, reduction='mean')

            pseudo_label = torch.softmax(logits_unlabel.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.95).float()
            we_can_believe = [index for index, value in enumerate(mask) if value == 1]
            if len(we_can_believe) != 0:
                classifier_loss_unlabel_batch = F.cross_entropy(logits_unlabel[we_can_believe], targets_u[we_can_believe], reduction='mean')
            if len(we_can_believe) == 0:
                classifier_loss_unlabel_batch = 0.0 * classifier_loss_label_batch

        if epoch % 2 == 0:
            proxy_anchor_loss_label_batch = loss_ProxyAnchor(feature_label, target.squeeze())
            proxy_anchor_loss_unlabel_batch = 0.0 * loss_ProxyAnchor(feature_label, target.squeeze())
            if epoch >= 100:
                if len(we_can_believe) >= 2:
                    proxy_anchor_loss_unlabel_batch = loss_ProxyAnchor(feature_unlabel[we_can_believe], targets_u[we_can_believe].squeeze())
            result_loss_batch, weight_pa = awl1(classifier_loss_label_batch, classifier_loss_unlabel_batch, proxy_anchor_loss_label_batch, proxy_anchor_loss_unlabel_batch)
            result_loss_batch.backward()
            optimizer1.step()

            classifier_loss_label += classifier_loss_label_batch.item()
            classifier_loss_unlabel += classifier_loss_unlabel_batch.item()
            proxy_anchor_loss_label += proxy_anchor_loss_label_batch.item()
            proxy_anchor_loss_unlabel += proxy_anchor_loss_unlabel_batch.item()
            result_loss += result_loss_batch.item()

        if epoch % 2 != 0:
            vat_loss_label_batch = vat_loss_function(data)
            vat_loss_unlabel_batch = vat_loss_function(data_vat)
            result_loss_batch, weight_vat = awl2(classifier_loss_label_batch, classifier_loss_unlabel_batch, vat_loss_label_batch, vat_loss_unlabel_batch)
            result_loss_batch.backward()
            optimizer2.step()
            classifier_loss_label += classifier_loss_label_batch.item()
            classifier_loss_unlabel += classifier_loss_unlabel_batch.item()
            vat_loss_label += vat_loss_label_batch.item()
            vat_loss_unlabel += vat_loss_unlabel_batch.item()
            result_loss += result_loss_batch.item()

        classifier_output = F.log_softmax(logits_label, dim=1)
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    if epoch % 2 == 0:
        classifier_loss_label /= len(train_dataloader)
        classifier_loss_unlabel /= len(train_dataloader)
        proxy_anchor_loss_label /= len(train_dataloader)
        proxy_anchor_loss_unlabel /= len(train_dataloader)
        result_loss /= len(train_dataloader)
        print(
            'Train Epoch: {} \tClassifier_Loss (Label): {:.6f}, Classifier_Loss (Unlabel): {:.6f}, Proxy Anchor Loss (Label): {:.6f}, Proxy Anchor Loss (Unlabel): {:.6f}, Combined_Loss :{:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
                epoch,
                classifier_loss_label,
                classifier_loss_unlabel,
                proxy_anchor_loss_label,
                proxy_anchor_loss_unlabel,
                result_loss,
                correct,
                len(train_dataloader.dataset),
                100.0 * correct / len(train_dataloader.dataset))
        )

    if epoch % 2 != 0:
        classifier_loss_label /= len(train_dataloader)
        classifier_loss_unlabel /= len(train_dataloader)
        vat_loss_label /= len(train_dataloader)
        vat_loss_unlabel /= len(train_dataloader)
        result_loss /= len(train_dataloader)
        print(
            'Train Epoch: {} \tClassifier_Loss (Label): {:.6f}, Classifier_Loss (Unlabel): {:.6f}, VAT Loss (Label): {:.6f}, VAT Loss (Unlabel): {:.6f}, Combined_Loss :{:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
                epoch,
                classifier_loss_label,
                classifier_loss_unlabel,
                vat_loss_label,
                vat_loss_unlabel,
                result_loss,
                correct,
                len(train_dataloader.dataset),
                100.0 * correct / len(train_dataloader.dataset))
        )


    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Combined_Loss/train', result_loss, epoch)
    writer.add_scalar('Classifier_Loss/train', classifier_loss_label, epoch)


def test(model, loss, test_dataloader, epoch, writer, device_num):
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

    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/validation', test_loss,epoch)

    return test_loss

def train_and_test(model, loss_function, loss_ProxyAnchor, train_dataset, VAT_dataset, val_dataset, optimizer1, awl1, optimizer2, awl2, epochs, writer, save_path, device_num, batch_size, vat_batch_size):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        VAT_dataloader = DataLoader(VAT_dataset, batch_size=vat_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        train(model, loss_function, loss_ProxyAnchor, train_dataloader, VAT_dataloader, optimizer1, awl1, optimizer2, awl2, epoch, writer, device_num)
        test_loss = test(model, loss_function, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        #     writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

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
        batch_size: int = 32,
        test_batch_size: int = 32,
        vat_batch_size: int = 250,
        epochs: int = 300,
        lr: float = 0.001,
        lr_prox: float = 0.05,
        log_interval: int = 10,
        n_classes: int = 10,
        ft: int = 62,
        save_path: str = 'model_weight/CNN_MAT1_n_classes_10_10label_90unlabel_rand30_Split3_autoweight100.pth',
        device_num: int = 0,
        rand_num: int = 30,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.vat_batch_size = vat_batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_prox = lr_prox
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.ft = ft
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num

def main():
    conf = Config()
    writer = SummaryWriter("logs_CNN_MAT1_n_classes_10_10label_90unlabel_rand30_Split3_autoweight100")
    device = torch.device("cuda:"+str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, conf.rand_num)

    train_dataset = TensorDataset(torch.Tensor(X_train_labeled), torch.Tensor(value_Y_train_labeled))
    VAT_dataset = TensorDataset(torch.Tensor(X_train_unlabeled), torch.Tensor(value_Y_train_unlabeled))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))

    model = base_complex_model()
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)

    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)

    loss_ProxyAnchor = Proxy_Anchor(nb_classes = conf.n_classes, sz_embed = 128, mrg = 0.1, alpha = 32)
    if torch.cuda.is_available():
        loss_ProxyAnchor = loss_ProxyAnchor.to(device)

    awl1 = AutomaticWeightedLoss(4)

    optim1 = torch.optim.Adam([
        {'params': model.parameters(), 'lr': conf.lr},
        {'params': awl1.parameters(), 'weight_decay': 0},
        {'params': loss_ProxyAnchor.parameters(), 'lr': conf.lr_prox}
    ])

    awl2 = AutomaticWeightedLoss(4)
    optim2 = torch.optim.Adam([
        {'params': model.parameters(), 'lr': conf.lr},
        {'params': awl2.parameters(), 'weight_decay': 0}
    ])


    train_and_test(model,
                   loss_function=loss,
                   loss_ProxyAnchor = loss_ProxyAnchor,
                   train_dataset=train_dataset,
                   VAT_dataset=VAT_dataset,
                   val_dataset=val_dataset,
                   optimizer1=optim1,
                   awl1=awl1,
                   optimizer2=optim2,
                   awl2=awl2,
                   epochs=conf.epochs,
                   writer=writer,
                   save_path=conf.save_path,
                   device_num=conf.device_num,
                   batch_size=conf.batch_size,
                   vat_batch_size=conf.vat_batch_size)

if __name__ == '__main__':
   main()