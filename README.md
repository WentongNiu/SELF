# SELF
A Multi-Task Learning Framework for Robust Person-Centric Relation Extraction

## Dataset
The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/).
```
 |-- data
 |    |-- tacred
 |    |    |-- train.txt       
 |    |    |-- val.txt
 |    |    |-- test.txt
 |    |    |-- rel2id.json
```

## Model
```
 |-- pretrain_model
 |    |-- roberta_base
```

## Training and Evaluation
To train and evaluate, run
```bash
Code/train.py
```

## Acknowledgement
Our code is based on [this repo](https://github.com/wzhouad/RE_improved_baseline) of the following paper.
```
@inproceedings{zhou2022improved,
  title={An Improved Baseline for Sentence-level Relation Extraction},
  author={Zhou, Wenxuan and Chen, Muhao},
  booktitle={Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  year={2022}
}
```
