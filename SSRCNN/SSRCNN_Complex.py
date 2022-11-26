import torch
from  torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(512)
        self.linear2 = nn.LazyLinear(128)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        embedding = F.relu(x)


        return embedding

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.LazyLinear(1152)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.zeropad1 = nn.ReplicationPad1d(2)
        self.conv1 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.zeropad2 = nn.ReplicationPad1d(3)
        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)

        self.upsample3 = nn.Upsample(scale_factor=2)
        self.zeropad3 = nn.ReplicationPad1d(2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)

        self.upsample4 = nn.Upsample(scale_factor=2)
        self.zeropad4 = nn.ReplicationPad1d(3)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)

        self.upsample5 = nn.Upsample(scale_factor=2)
        self.zeropad5 = nn.ReplicationPad1d(2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)

        self.upsample6 = nn.Upsample(scale_factor=2)
        self.zeropad6 = nn.ReplicationPad1d(4)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)

        self.upsample7 = nn.Upsample(scale_factor=2)
        self.zeropad7 = nn.ReplicationPad1d(4)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)

        self.upsample8 = nn.Upsample(scale_factor=2)
        self.zeropad8 = nn.ReplicationPad1d(2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)

        self.upsample9 = nn.Upsample(scale_factor=2)
        self.zeropad9 = nn.ReplicationPad1d(3)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)

        self.conv10 = ComplexConv(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self,x):
        x = self.linear1(x)
        x = x.view(-1, 128, 9)

        x = self.upsample1(x)
        x = self.zeropad1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.upsample2(x)
        x = self.zeropad2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.upsample3(x)
        x = self.zeropad3(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        x = self.upsample4(x)
        x = self.zeropad4(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)

        x = self.upsample5(x)
        x = self.zeropad5(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)

        x = self.upsample6(x)
        x = self.zeropad6(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)

        x = self.upsample7(x)
        x = self.zeropad7(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)

        x = self.upsample8(x)
        x = self.zeropad8(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)

        x = self.upsample9(x)
        x = self.zeropad9(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)

        x = self.conv10(x)
        x = F.sigmoid(x)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(128,10)

    def forward(self,x):
        x = self.linear(x)
        return x

class SR2CNN(nn.Module):
    def __init__(self):
        super(SR2CNN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()

    def forward(self, x):
        features = self.encoder(x)
        r_x = self.decoder(features)
        pre = self.classifier(features)

        return features, r_x, pre

class SSRCNN(nn.Module):
    def __init__(self):
        super(SSRCNN, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def forward(self, x):
        feature = self.encoder(x)
        x = self.classifier(feature)

        return feature, x

if __name__ == "__main__":
    model = SRRCNN()
    input = torch.randn((10,2,6000))

    input_noise = input + torch.randn((input.shape[0],input.shape[1],input.shape[2]))
    output = model(input)
    output_noise = model(input_noise)


