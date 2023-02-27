# Semi-Supervised Specific Emitter Identification Method Using Metric-Adversarial Training
The code corresponds to the paper https://arxiv.org/abs/2211.15379 or https://ieeexplore.ieee.org/document/10026879

# Requirement
pytorch 1.10.2
python 3.6.13

# Framework of MAT
![Framework of MAT](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/MAT_Pipeline.pdf)

# Dataset
We use the dataset proposed in paper [55] and [56] to evaluate our proposed MAT-based SS-SEI method. The former is a large-scale real-world radio signal dataset based on a special aeronautical monitoring system, ADS-B, and the latter is WiFi dataset collected from USRP X310 radios that emit IEEE 802.11a standards compliant frames. The number of categories of ADS-B dataset and WiFi dataset is 10 and 16, respectively. The length of each sample of ADS-B dataset and WiFi dataset is 4,800
and 6,000, respectively. The number of training samples of ADS-B dataset and WiFi datsset is 3, 080. The number of testing samples of ADS-B dataset and WiFi dataset is 1,000 and 16,004, respectively. We construct five semi-supervised scenarios and one fully supervised scenario, where the number of labeled training samples to the number of all training samples ratio is {5%, 10%, 20%, 50%, 100%}, to evaluate the identification performance of the proposed SS-SEI method. In addition, 30% of the training samples is used as the validating samples during the training process.

[55] Y. Tu, Y. Lin, et al., “Large-scale real-world radio signal recognition with deep learning,” Chin. J. Aeronaut., vol. 35, no. 9, pp. 35–48, Sept.
2022.

[56] K. Sankhe, M. Belgiovine, F. Zhou, S. Riyaz, S. Ioannidis, and K. Chowdhury, “ORACLE: Optimized radio classification through convolutional neural networks,” in IEEE Conf. Comput. Commun., Apr.2019, pp. 370-378.

The dataset can be downloaded from the Link: https://pan.baidu.com/s/13qW5mnfgUHBvWRid2tY2MA Passwd：eogv

# Classification Accuracy
 Methods  | ADS-B (5%) | ADS-B (10%) | WiFi (5%) | WiFi (10%)
 ---- | ----- | ------  | ----- | ------  |
 CVNN  | 60.50% |  74.50% | 20.47% |28.64%
 DRCN  | 54.20% | 72.40% | 21.94% | 47.51%
 SSRCNN | 49.30% | 79.30% | 19.33% | 38.09%
 TripleGAN | 45.10% | 61.10% | 27.57% | 37.27%
 SimMIM | 65.90% | 77.90% | 31.71% | 49.59%
 MAT-CL | 70.06% | 83.80% | 27.26% | 80.70%
 MAT-PA | 74.00% | 84.80% | 28.82% | 54.96%

# Features Visualization
![Features Visualization of CNN](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/CNN_n_classes_16_10label_90unlabel_improved.png)

CVNN

![Features Visualization of DRCN](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/DRCN_complex_n_classes_16_10label_90unlabel_improved.png)

DRCN

![Features Visualization of SSRCNN](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/SSRCNN_n_classes_16_10label_90unlabel_improved.png)

SSRCNN

![Features Visualization of TripleGAN](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/TripleGAN_n_classes_16_10label_90unlabel_improved.png)

TripleGAN

![Features Visualization of SimMIM](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/SimMIM_encoder_mask05_n_classes_16_label10_improved.png)

SimMIM

![Features Visualization of MAT](https://github.com/lovelymimola/MAT-based-SS-SEI/blob/master/Figures/CNN_MAT_n_classes_16_10label_90unlabel_improved.png)

MAT

# E-mail
If you have any question, please feel free to contact us by e-mail (1020010415@njupt.edu.cn).
