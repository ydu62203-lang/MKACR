# MKACR on RecBole

This repository contains the official implementation for the paper **"Multi-Space Knowledge Graph Embedding and Adversarial Contrastive Learning for Recommendation"**.

Our model, **MKACR**, is developed based on the comprehensive recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

[Paper (Link to be added)] | [Code Repository](https://github.com/ydu62203-lang/recbole)

---

## Overview

**MKACR** is a novel knowledge-aware recommendation model that leverages multi-space knowledge graph embeddings and an adversarial contrastive learning strategy. By projecting entities and relations into distinct spaces, our model captures complex semantics within the knowledge graph. The adversarial contrastive learning component ensures that the embeddings are robust and highly discriminative, leading to improved recommendation performance.

This implementation uses the powerful and flexible RecBole framework, which provides a unified structure for data processing, model training, and evaluation. We have integrated MKACR as a new knowledge-aware recommender and adapted parts of the training process to support its unique adversarial mechanism.

<p align="center">
  <img src="asset/framework.png" alt="RecBole v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: The Overall Architecture of the RecBole Framework
</p>

## Installation

Our code requires Python 3.7 or later and PyTorch 1.7.0 or later. Please ensure your environment meets these requirements.

### 1. Clone the Repository
Clone this repository which contains the MKACR model and the modified RecBole framework.
```bash
git clone [https://github.com/ydu62203-lang/recbole.git](https://github.com/ydu62203-lang/recbole.git)
cd recbole
```

### 2. Install Dependencies
Install all the required packages, including RecBole in editable mode. This allows you to use the MKACR model seamlessly.
```bash
pip install -e . --verbose
```

## Quick Start: Running MKACR

Follow these steps to train and evaluate the MKACR model on a sample dataset like `ml-1m`.

### 1. Prepare the Dataset
RecBole will automatically download and process the dataset if it's not found in the `dataset/` directory.

### 2. Create a Configuration File
Create a YAML configuration file to specify the model and its hyperparameters. For example, create a file named `run_mkacr.yaml` in the root directory with the following content.

**`run_mkacr.yaml`:**
```yaml
# Model and dataset configuration
model: MKACR
dataset: ml-1m

# Training and evaluation settings
stopping_step: 10
epochs: 100
train_batch_size: 4096
eval_batch_size: 4096
load_col:
    inter: [user_id, item_id, rating, timestamp]
    kg: [head_id, relation_id, tail_id]

# MKACR-specific hyperparameters
embedding_size: 64
reg_weight: 1.e-5
context_hops: 2

# For adversarial training (if applicable)
adv_weight: 0.1

# Evaluation metrics
metrics: [Recall, MRR, NDCG, Hit, Precision]
topk: [10, 20]
valid_metric: MRR@10
```

### 3. Run the Model
Execute the following command from the repository's root directory. The script will use the model (`MKACR`), dataset (`ml-1m`), and custom parameters defined in your YAML file.

```bash
python run_recbole.py --config_files=run_mkacr.yaml
```

The training process will begin, and you will see output similar to the following:
```
INFO ml-1m
The number of users: 6041
The number of items: 3707
...
INFO MKACR(
  (user_embedding): Embedding(6041, 64)
  (item_embedding): Embedding(3707, 64)
  ...
)
Trainable parameters: XXXXXX
INFO epoch 0 training [time: X.XXs, train loss: X.XXXX]
INFO epoch 0 evaluating [time: X.XXs, valid_score: X.XXXX]
INFO valid result:
recall@10: X.XXXX  mrr@10: X.XXXX  ndcg@10: X.XXXX  hit@10: X.XXXX  precision@10: X.XXXX
...
INFO Finished training, best eval result in epoch XX
INFO best valid result:
recall@10: X.XXXX  mrr@10: X.XXXX  ndcg@10: X.XXXX  hit@10: X.XXXX  precision@10: X.XXXX
INFO test result:
recall@10: X.XXXX  mrr@10: X.XXXX  ndcg@10: X.XXXX  hit@10: X.XXXX  precision@10: X.XXXX
```

## Implementation Details

To implement the MKACR model, we made the following key modifications to the standard RecBole framework:
* **`recbole/model/knowledge_aware_recommender/mkacr.py`**: This file contains the core implementation of the MKACR model architecture. It is built upon the KGAT model as a structural baseline.
* **`recbole/trainer/trainer.py`**: The trainer has been slightly modified to accommodate the adversarial contrastive learning loop required by our model.
* **`recbole/properties/model/MKACR.yaml`**: Default hyperparameters for the MKACR model are defined here. These can be overridden using a custom configuration file as shown in the Quick Start section.

## Cite

If you find our work useful for your research, please consider citing our paper.
```bibtex
@inproceedings{your-lastname-202X-mkacr,
  author    = {Author Name},
  title     = {Multi-Space Knowledge Graph Embedding and Adversarial Contrastive Learning for Recommendation},
  booktitle = {Conference Name},
  year      = {202X}
}
```

Please also cite the original **RecBole** papers, as our work is built upon their excellent library.
```bibtex
@inproceedings{recbole[1.0],
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}

@inproceedings{recbole[2.0],
  author    = {Wayne Xin Zhao and Yupeng Hou and Xingyu Pan and Chen Yang and Zeyu Zhang and Zihan Lin and Jingsen Zhang and Shuqing Bian and Jiakai Tang and Wenqi Sun and Yushuo Chen and Lanling Xu and Gaowei Zhang and Zhen Tian and Changxin Tian and Shanlei Mu and Xinyan Fan and Xu Chen and Ji{-}Rong Wen},
  title     = {RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  booktitle = {{CIKM}},
  pages     = {4722--4726},
  publisher = {{ACM}},
  year      = {2022}
}
```

## License
This project is licensed under the [MIT License](./LICENSE).

## Acknowledgments
This project would not be possible without the foundational work of the [RecBole Team](https://recbole.io/about.html). We sincerely thank them for developing and maintaining this comprehensive library.
