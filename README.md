# Pyramid Super-Resolution

This repos contains source code for [paper](https://ieeexplore.ieee.org/abstract/document/9143068):

T. Vo, G. Lee, H. Yang and S. Kim, "Pyramid With Super Resolution for In-the-Wild Facial Expression Recognition," in IEEE Access, vol. 8, pp. 131988-132001, 2020, doi: 10.1109/ACCESS.2020.3010018.


## Data
The datasets are place in data folder, we do not include data with repos, you should prepare it yourself by correct 
structure, please see [data readme](data/readme.txt).

RAF-DB dataset, see [meta for RAF-DB](data/raf-db/raf-db-meta.csv.png), *5_fold* randomly, to separated train/valid.

## Models
The architecture uses EDSR as base super-resolution.

A pretrain for EDSR (x2) could be found at [https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt](https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt),
the original source [colorjam/EDSR-PyTorch](https://github.com/colorjam/EDSR-PyTorch/blob/master/src/model/edsr.py).
Please download and copy to *models/super_resolution* folder.

In case of transfer learning from FER+ to RAF-DB, download [final.w](https://drive.google.com/file/d/1zHB7lKXqbN2np-cXdMMVNOFsWEnE494R/view?usp=sharing) and place in *models/ferplus/vgg16_bn_quick_3size_e20*.
Using *[ferplus-vgg16-3size.json](config/ferplus-vgg16-3size.json)* if you want make it yourself.

## Run
Setup environment:
    
    pip install -r requirements.txt
    
Note: this work with pytorch 1.6 and fastai version < 1.x (not working with fastai 2.x).

Command to train/test:
        
    python -m prlab.cli k_fold --json_conf config/raf-db.json 

raf-db.json is a configure file which some field as below:
- base_weights_path: to transfer learning, e.g. from FER+ (optional, weights provided, or you can use configure ferplus-vgg16-3size to
train and get this weights)
- weight_path_x2: edsr pretrain (optional)

report will be in *[models/*/reports.txt](models/raf-db/reports.txt)*.

Source code is provided 'as-is' WITHOUT any WARRANTY or SUPPORT. Using this script is at YOUR OWN RISK.
