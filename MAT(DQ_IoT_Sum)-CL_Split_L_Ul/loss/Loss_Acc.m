clc
clear all
close all
x=1:299;
temp = importdata('run-.-tag-Accuracy_train.csv');
train_acc = temp.data(x,3);

temp = importdata('run-.-tag-Classifier_Loss_train.csv');
train_loss = temp.data(x,3);

temp = importdata('run-.-tag-Accuracy_validation.csv');
val_acc = temp.data(x,3);

temp = importdata('run-.-tag-Classifier_Loss_validation.csv');
val_loss = temp.data(x,3);
x=x';
yyaxis left
train_acc_=plot(x,train_acc,'b-','LineWidth',2,'MarkerSize',8);
hold on
test_acc_=plot(x,val_acc,'b--','LineWidth',2,'MarkerSize',8);
hold on

yyaxis right
train_loss_=plot(x,train_loss,'r-','LineWidth',2,'MarkerSize',8);
hold on
test_loss_=plot(x,val_loss,'r--','LineWidth',2,'MarkerSize',8);
hold on

legend([train_acc_,train_loss_,test_acc_,test_loss_],...
    'Training accuracy',...
    'Training loss',...
    'Validation accuracy',...
    'Validation loss',...
    'location','Best')
xlabel('Epoch Times');
yyaxis left;
ylabel('Accuracy');
yyaxis right
ylabel('Loss')
set(gca,'FontSize',16,'FontName','Times New Roman');
grid on