# AIR-MDSAR-Map
**Please go to "[AIR-MDSAR-Map](https://nairongzheng.github.io/AIR-MDSAR-Map/)" for details.**

## 1. Data preprocessing

(AIR-MDSAR-Map is generated after Data preprocessing)

### (1) SAR and optical images
In order to facilitate training, the white edge of the original image is changed to black edge: 

- Extracting the position of white edge: [code](https://github.com/NairongZheng/utils/blob/main/find_edge.py).
- changing white edge to black: [python code](https://github.com/NairongZheng/utils/blob/main/delete_edge.py), [matlab code](https://github.com/NairongZheng/utils/blob/main/delete_edge.m).

You can also view the distribution of intensity values for each channel of the images: [code](https://github.com/NairongZheng/utils/blob/main/plot_histogram.py).

### (2) Label generating

labels are generated manually using matlab, secondary labeling can refer to: [code](https://github.com/NairongZheng/utils/blob/main/imageLabel_change_path.m).

Semi-automatic annotation algorithm: [paper](https://link.springer.com/chapter/10.1007/978-981-19-8202-6_9).

converting the labels to RGB images: [code](https://github.com/NairongZheng/utils/blob/main/changelabel_123_imageLabeler.py).

Calculating the proportion of categories in labels: [code](https://github.com/NairongZheng/utils/blob/main/plot_pie.py).

Overlap the label with the image to check the accuracy of the annotation: [code](https://github.com/NairongZheng/utils/blob/main/image_with_mask.py).

### (3) Generating AIR-MDSAR-Map

Generating AIR-MDSAR-Map: [code](https://github.com/NairongZheng/utils/blob/main/gen_public_dataset.py)

AIR-MDSAR-Map contains polarization SAR images in bands of P, L, S, C, and Ka and high-resolution optical images in Wanning, Hainan, and Sheyang, Jiangsu. Land cover objects are divided into water, bare soil, road, industry, vegetation, residence, plantation, farms, and other. Each class is labeled with different colors, i.e. water in the blue, vegetation in green.

**Please go to "[AIR-MDSAR-Map](https://nairongzheng.github.io/AIR-MDSAR-Map/)" for details.**

You can stitch the small patches back to the original size, refer to [code](https://github.com/NairongZheng/utils/blob/main/connecting_images.py). 

## 2. Classical algorithm verification

1. [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), [SegNet](https://ieeexplore.ieee.org/document/7803544), [DeeplabV3p](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49), and [HRNet](https://ieeexplore.ieee.org/document/9052469) are used for the validation of AIR-MDSAR-Map: [code(Tensorflow)](https://github.com/NairongZheng/AIR-MDSAR-Map/tree/main/s1_classical), other algorithms can refer to [paper](https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-16/issue-1/014520/Land-cover-classification-of-synthetic-aperture-radar-images-based-on/10.1117/1.JRS.16.014520.short).
2. Evaluation index calculation can be referred to [code](https://github.com/NairongZheng/utils/blob/main/cal_ConfusionMatrix_indicators_2.py).

## 3. Land cover classification

Generating train and test data: [python code](https://github.com/NairongZheng/utils/blob/main/generate_test_dataset.py), [matlab code](https://github.com/NairongZheng/utils/blob/main/generate_test_dataset_x.m).

Cutting images to patches: [code](https://github.com/NairongZheng/utils/blob/main/cutting_images_2.py).

### (1) Single band classification

Single band classification: [code](https://github.com/NairongZheng/AIR-MDSAR-Map/tree/main/s2_yyjf).

### (2) Fusion classification

1. Priori fusion: [code](https://github.com/NairongZheng/utils/blob/main/results_fusion.py).
2. Model fusion: [code](https://github.com/NairongZheng/AIR-MDSAR-Map/tree/main/s2_yyjf).
3. Feature fusion: [code](https://github.com/NairongZheng/AIR-MDSAR-Map/tree/main/s3_fusion).

