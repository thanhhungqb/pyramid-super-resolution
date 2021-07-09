# Pyramid Super-Resolution

This repos contains source code for [paper](https://ieeexplore.ieee.org/abstract/document/9143068):

T. Vo, G. Lee, H. Yang and S. Kim, "Pyramid With Super Resolution for In-the-Wild Facial Expression Recognition," in IEEE Access, vol. 8, pp. 131988-132001, 2020, doi: 10.1109/ACCESS.2020.3010018.


## Data
The datasets are placed in the data folder, we do not include data with repos, you should prepare it yourself by correct 
structure, please see [data readme](data/readme.txt).

RAF-DB dataset, see [meta for RAF-DB](data/raf-db/raf-db-meta.csv.png), *5_fold* randomly, to separated train/valid.

## Models
The architecture uses EDSR as base super-resolution.
A pretrain for EDSR (x2) could be found at [https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt](https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt), the original source [colorjam/EDSR-PyTorch](https://github.com/colorjam/EDSR-PyTorch/blob/master/src/model/edsr.py). Please download and copy to *models/super_resolution* folder (weight_path_x2). *(disclaimer: EDSR and weights are not our)*

In case of transfer learning from FER+ to RAF-DB, download [final.w](https://drive.google.com/file/d/1zHB7lKXqbN2np-cXdMMVNOFsWEnE494R/view?usp=sharing) and place in *models/ferplus/vgg16_bn_quick_3size_e20* (edit base_weights_path in configure file). Using *[ferplus-vgg16-3size.json](config/ferplus-vgg16-3size.json)* if you want make it yourself.

## Note about configure file (.json)
The configure file includes many parameters to train/test the network architecture. Most of them named with easy to understand, and you should read and understand yourself. There are some highlights:
- *lr* has an important role when training our network architecture. It should be separated into 3 increment values. If you use the same *lr* for all blocks, it is difficult to get an optimal model.
- weight_path_x2: EDSR pre-trained (x2) (similar for x3 and x4 if you use)
- base_weights_path: for pre-trained the base_arch, e.g., vgg16_bn, set named base_weights_path_NU for not use. (see note inside JSON). For FER+, it may help the training process, for RAF-DB it can be used to transfer learning, e.g. from FER+ (optional, weights provided, or you can use configure ferplus-vgg16-3size to train and get these weights).

## Run
Setup environment:
    
    pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
    
Note: **This source code works with PyTorch 1.6 and Fastai version 1.x (not working with fastai 2.x).**

### FER+ dataset

    python -m prlab.cli run --json_conf config/ferplus-psr-edsr-x2.json

Note: 
- pre-trained for 3 size for the base arch maybe help the network coverage quicker nut not nessary.

### RAF-DB dataset
Command to train/test:
        
    python -m prlab.cli k_fold --json_conf config/raf-db.json 


report will be in *[models/*/*reports.txt](models/raf-db/reports.txt)*.

## Popular Q&A
- Q: I got the low performance, far below the number report in the paper, e.g., FER+ only get ~85%. A: Please follow the experiment setup, the most important is *lr*, and for RAF-DB, the transfer from FER+ weights maybe help. After correctly setup, we could get *similar* our performance, sometimes a little better (not too far below is enough for DL).
- Q: Install requirements with errors. A: Please make sure the environment are correctly set up with some compiler for python developer such as gcc. After that, *"pip install --upgrade pip wheel setuptools"* maybe need some version of python. We recommend using virtualenv to set up the experiment's environment.

Source code is provided 'as-is' WITHOUT any WARRANTY or SUPPORT. Using this script is at YOUR OWN RISK.

