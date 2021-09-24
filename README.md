<h2 align="center">BEAR: Bilinear neural network for Efficient Approximation of RPCA</h2>

<p align="center">
<img width="65%" src="src/surveillance.gif">
<img width="65%" src="src/confocal_demo_compressed.gif">
</p>
<h6 align="center">Demo videos acquired using BEAR. Input video, low-rank component, and sparse component from left to right.</h6>

Official source codes for ["Efficient neural network approximation of robust PCA for automated analysis of calcium imaging data"](https://link.springer.com/chapter/10.1007%2F978-3-030-87234-2_56), MICCAI 2021  [[1]](#ref).

There is also **[Matlab implementation](https://drive.google.com/file/d/1PqAhtM9712tQme_vP5GvJ6mSWoogbxRF/view?usp=sharing) (~16MB) of BEAR.**

Source codes for PCP, IALM, and GreGoDec are from [lrslibrary](https://github.com/andrewssobral/lrslibrary).
Source codes for OMWRPCA are from [onlineRPCA](https://github.com/wxiao0421/onlineRPCA/blob/master/rpca/omwrpca.py).
(Both last visited on March, 2021.)

## Running BEAR (in jupyter notebook)

In `demo.ipynb` notebook, you can follow how to use BEAR for surveillance video. There is a video saved in this notebook, where they may not visible if you simply open the notebook in github. One way to view all things in the notebook is opening it with VS Code.

Following is the visualization result in the notebook. You can see that low rank `L` and sparse `S` is well decomposed from input video `Y`.
<p align="center">
<img width="65%" src="src/demo_hall.gif">
</p>

## Running BEAR (in local)

You can run BEAR by very simple code. Just download or clone this repository and use the scripts below.
`--D` refers the name of the data and `--d` enables the default settings of several hyperparameters.
There are already two videos in `data` folder: small surveillance video, and calcium imaging data of mouse which are widely used in RPCA paper. You can first try with these data. You can try BEAR with as follows:
```bash
python scripts/run_BEAR.py --D hall --d True
python scripts/run_BEAR.py --D demoMovie --d True
```

Also you can download **additional** calcium imaging data of zebrafish in this [Google Drive](https://drive.google.com/file/d/1gQRJzk5rR5TRYc5O_zZDsuXRK6tAEBPZ/view?usp=sharing) (~25MB). If you want to see more examples, try these. Download, unzip, and move .tif and .mat files inside the `data` folder. Then, you can try BEAR with as follows:
```bash
python scripts/run_BEAR.py --D confocal_zebrafish --d True
python scripts/run_BEAR.py --D confocal_zebrafish_2 --d True
```
Results will be automatically saved in `results/BEAR/(name of the data)_(Y/L/S)`.

## Hardware specification & Requirements

**Hardware specification**

```markdown
OS : Ubuntu 18.0.4
CPU : Intel i7-9700K
GPU : GeForce RTX 2080 Ti 11GB
RAM : 128GM
```

**Requirements (May be okay with different versions.)**
```markdown
torch==1.7.0
numpy==1.19.4
psutil==5.7.2

scipy==1.5.3
scikit-image==0.17.2
mat73==0.46
```

## Reproduce the paper

Calcium imaging data of zebrafish we have used in paper is in this [Google Drive](https://drive.google.com/file/d/115lCnwIVU0TtKedQ_31FDaOG8wksGmG9/view?usp=sharing) (~10GB). They are very large to show that BEAR is fast and scalable to use in large calcium imaging data.

1. Figure 3. Phase diagram.
```bash
python scripts/phase_diagram.py --D None --d None
```

2. Table 1. and Figure 4. Decomposition of the zebrafish caclium imaging data.
```bash
python scripts/run_Greedy_BEAR.py --D zebrafish_150 --d True
python scripts/run_Greedy_BEAR.py --D zebrafish_1000 --d True
```
Due to the size of data, loading files itself does take long time (Minutes in HDD).
And for 1000 length video, about 100GB of RAM is required.

3. Figure 5. and Figure 6. Cascaded BEAR for analysis of neuronal activity.
```bash
python scripts/run_Cascaded_BEAR.py --D demoMovie --d True
python scripts/run_Cascaded_BEAR.py --D spinning_confocal --d True
```
For accuracy, number of epochs in default settings of Cascaded BEAR is very large.
Can be observed that loss value does not decrease actually after small number of epochs.
You can reduce the `args.epoch` if you want.

## Citation
<a name="ref"></a>
```markdown
@InProceedings{han2021efficient,
author="Han, Seungjae and Cho, Eun-Seo and Park, Inkyu and Shin, Kijung and Yoon, Young-Gyu",
title="Efficient Neural Network Approximation of Robust PCA for Automated Analysis of Calcium Imaging Data",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="595--604",
url="https://link.springer.com/chapter/10.1007%2F978-3-030-87234-2_56"
}
```
