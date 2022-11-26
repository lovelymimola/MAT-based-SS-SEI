import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset_10label import *
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sklearn.metrics as sm
from sklearn import manifold

def scatter(features, targets, subtitle = None, n_classes = 10):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
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

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)


def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train, X_val, value_Y_train_labeled, value_Y_train, value_Y_val = TrainDataset(n_classes, rand_num)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TestDataset_prepared(n_classes, rand_num):
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
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
    X_test, Y_test = TestDataset_prepared(10, rand_num=50)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load("model_weight/CNN_n_classes_10_10label_90unlabel_rand30.pth")
    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    scatter(eval_tsne_embeds, target.astype('int64'), "CNN_n_classes_10_10label_90unlabel", 10)
    visualize_data(X_test_embedding_feature_map, target.astype('int64'), "Visualization/CNN_n_classes_10_10label_90unlabel_improved", 10)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))

if __name__ == "__main__":
    main()