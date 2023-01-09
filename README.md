# AIR-MDSAR-Map
**Please go to "[AIR-MDSAR-Map](https://nairongzheng.github.io/AIR-MDSAR-Map/)" for details.**

## Data preprocessing

(AIR-MDSAR-Map is generated after Data preprocessing)

### SAR and optical images
In order to facilitate training, the white edge of the original image is changed to black edge: 

- Extracting the position of white edge: [python code](https://github.com/NairongZheng/utils/blob/main/find_edge.py)
- changing white edge to black: [python code](https://github.com/NairongZheng/utils/blob/main/delete_edge.py), [matlab code](https://github.com/NairongZheng/utils/blob/main/delete_edge.m)

You can also view the distribution of intensity values for each channel of the images: [code](https://github.com/NairongZheng/utils/blob/main/plot_histogram.py)

### Label generating

labels are generated manually using matlab, secondary labeling can refer to: [code](https://github.com/NairongZheng/utils/blob/main/imageLabel_change_path.m)

converting the labels to RGB images: [code](https://github.com/NairongZheng/utils/blob/main/changelabel_123_imageLabeler.py)

Calculating the proportion of categories in labels: [code](https://github.com/NairongZheng/utils/blob/main/plot_pie.py)

Overlap the label with the image to check the accuracy of the annotation: [code](https://github.com/NairongZheng/utils/blob/main/image_with_mask.py)

### Generating AIR-MDSAR-Map

Generating AIR-MDSAR-Map: [code](https://github.com/NairongZheng/utils/blob/main/gen_public_dataset.py)

AIR-MDSAR-Map contains polarization SAR images in bands of P, L, S, C, and Ka and high-resolution optical images in Wanning, Hainan, and Sheyang, Jiangsu. Land cover objects are divided into water, bare soil, road, industry, vegetation, residence, plantation, farms, and other. Each class is labeled with different colors, i.e. water in the blue, vegetation in green.

**Please go to "[AIR-MDSAR-Map](https://nairongzheng.github.io/AIR-MDSAR-Map/)" for details.**