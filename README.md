# THAN
Codes, datasets for paper "TemporalHAN: Hierarchical Attention-Based Heterogeneous Temporal Network Embedding"

![AppVeyor](https://img.shields.io/badge/python-3.6.13-blue)
![AppVeyor](https://img.shields.io/badge/numpy-1.19.5-red)
![AppVeyor](https://img.shields.io/badge/tensorflow-1.6.0-brightgreen)
![AppVeyor](https://img.shields.io/badge/keras-2.2.0-orange)

## Run Code

1 Download preprocessed data and modify data path, then:
```
python ex_acm3025.py
```

2 Perform data set processing:
```
python THAN_aminer_Y1.py
python THAN_aminer_Y2.py
python THAN_aminer_Y3.py
python THAN_aminer_build_real_new_Y1_mat.py
python THAN_aminer_build_real_new_Y2_mat.py
python THAN_aminer_build_real_new_Y3_mat.py
```
Other data sets are similar.


3 Select the corresponding data set and downstream task:
```
python THAN_aminer_ex_new_3yi.py
python THAN_aminer_RP.py
python THAN_yelp_ex_new_3yi.py
python THAN_yelp_RP_test.py
```

## Datasets
/data/tips.txt
