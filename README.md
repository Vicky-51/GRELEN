# GRELEN

Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning

This repository contains the implementation of 

*Zhang, W., Zhang, C., and Tsung, F. " GRELEN: Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning ," Proceedings of the 31st International Joint Conference on Artificial Intelligence and the 25th European Conference on Artificial Intelligence (IJCAI-ECAI 2022)* [[PDF]](https://www.ijcai.org/proceedings/2022/0332.pdf)

### Datasets

Four datasets are included in our paper:

SWAT and WADI: [[download-from-iTrust]]((https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi))

SMD: [[download-from-OmniAnomaly]](https://github.com/smallcowbaby/OmniAnomaly)

PSM: [[download-from-RANSynCoders]]((https://github.com/eBay/RANSynCoders))

### Getting Started

* For training:

```
python train_grelen.py
```

* For testing:

```
python test_grelen.py
```

The config files for datasets, model and evaluation should be put in the config_files folder. Config files should contain the data path, model hyper-parameters, training settings for model training. Target model (parameter path) should be included while model testing.
