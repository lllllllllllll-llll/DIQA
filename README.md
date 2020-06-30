# DIQA
Pytorch version of IEEE Transactions on Image Processing 2019 : [J. Kim, A. Nguyen and S. Lee, "Deep CNN-Based Blind Image Quality Predictor," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 1, pp. 11-24, Jan. 2019, doi: 10.1109/TNNLS.2018.2829819.](https://ieeexplore.ieee.org/document/8383698)

# Note
1. Some training details differ from the original paper, if you want to be consistent with the original pape, make some changes.
2. This training progress only support on LIVE II database now, the training progress on TID2013, CSIQ, LIVEMD, CLIVE will be released soon.

# Train
1. For the Step 1 training, run `python train_step1.py`
1. For the Step 2 training, run `python train_step2.py`

# TODO
* Cross dataset test code will be published
* Train on different distortion types on LIVE, TID2013, CSIQ will be published
* Code of evaluations on Waterloo Exploration Database (D-test, L-test, P-test and gMAd competition) will be published
