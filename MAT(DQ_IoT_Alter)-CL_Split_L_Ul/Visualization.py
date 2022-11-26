import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset_15label import *

def scatter(features, targets, subtitle = None, n_classes = 10):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)

def Data_prepared(n_classes):
    X_train_labeled, X_train_unlabeled, X_train, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_train, value_Y_val = TrainDataset(n_classes)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TestDataset_prepared(n_classes):
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    return X_test, Y_test

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:2")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output[0])-1] = output[0].tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def main():
    X_test, Y_test = TestDataset_prepared(10)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load('model_weight/CNN_MAT_n_classes_10_15label_85unlabel.pth')
    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    scatter(eval_tsne_embeds, target.astype('int64'), "CNN_MAT_n_classes_10_15label_85unlabel", 10)

if __name__ == "__main__":
    main()