<h2 align="center">BEAR: Bilinear neural network for Efficient Approximation of RPCA</h2>

<p align="center">
<img width="65%" src="src/confocal_demo_compressed.gif">
</p>
<h6 align="center">Demo video acquired using BEAR. Input video, low-rank component, and sparse component from left to right. 
  
  This data is not included in the paper and this repository.</h6>

Official source codes for "Efficient neural network approximation of robust PCA for automated analysis of calcium imaging data", MICCAI 2021.

Source codes for PCP, IALM, and GreGoDec are from [lrslibrary](https://github.com/andrewssobral/lrslibrary).
Source codes for OMWRPCA are from [onlineRPCA](https://github.com/wxiao0421/onlineRPCA/blob/master/rpca/omwrpca.py).
(Both last visited on March, 2021.)


## Running BEAR

You can run BEAR which does Robust PCA by very simple code.
Just download or clone this repository and use the code below.
`--D` refers the name of the data and `--d` enables the default settings of several hyperparameters.

```bash
python scripts/run_BEAR.py --D hall --d True
```

Results will be saved in `results/BEAR/hall_(Y/L/S)` directory.

## Hardware specification & Requirements

**Hardware specification**

```markdown
CPU : Intel Xeon Silver 4214 CPU @ 2.20GHz
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

First of all, calcium imaging data we have used in paper is in this [Google Drive](www.google.com). Download, unzip, and move .tif and .mat files inside the data folder. We have already added small surveillance video which is widely used in RPCA paper. You can also try it.

1. Figure 3. Phase diagram.
```bash
python scripts/phase_diagram.py --D None --d None
```

2. Table 1. and Figure 4. Decomposition of the zebrafish caclium imaging data.
```bash
python scripts/run_Greedy_BEAR.py --D zebrafish_150 --d True (For 150 length video)
python scripts/run_Greedy_BEAR.py --D zebrafish_1000 --d True (For 1000 length video)
```
Due to the size of data, loading files itself does take long time. (Minutes in HDD)
And for 1000 length video, about 100GB of RAM is required.

3. Figure 5. and Figure 6. Cascaded BEAR for analysis of neuronal activity.
```bash
python scripts/run_Cascaded_BEAR.py --D demoMovie --d True
python scripts/run_Cascaded_BEAR.py --D spinning_confocal --d True
```
For accuracy and for safety, number of epochs in default settings of Cascaded BEAR is large.
Can be observed that loss value does not decrease actually after small number of epochs.
You can reduce the `args.epoch` if you want.


