# Official Implementation for "Multi-Space Knowledge Graph Embedding and Adversarial Contrastive Learning for Recommendation"

This repository contains the official source code and configuration files to reproduce the results for our paper, "Multi-Space Knowledge Graph Embedding and Adversarial Contrastive Learning for Recommendation".

Our proposed model, **MKACR**, is implemented within a modified version of the [RecBole](https://github.com/RUCAIBox/RecBole) framework. To facilitate reproducibility, we provide the complete source code.

## Code Structure

Our implementation involves modifications to several files within the RecBole library. The key components of our work are located in the following files:

* **1. Main Execution Script (`run_kgat.py`)**: The primary script to launch the training and evaluation of the MKACR model.
* **2. Model Implementation (`recbole/model/knowledge_aware_recommender/kgat.py`)**: This file contains the core implementation of our proposed **MKACR** model.
    > *Note: For seamless integration with the existing framework, the original `mkacr.py` file has been renamed to `kgat.py` to replace the base KGAT model.*
* **3. Custom Trainer (`recbole/trainer/trainer.py`)**: This file has been modified to include a custom `KGATTrainer` class, which handles the specific multi-task training loop required by MKACR (knowledge distillation, recommendation loss, etc.).
* **4. Model Hyperparameters (`recbole/properties/model/KGAT.yaml`)**: This YAML file defines the default hyperparameters for the MKACR model.
* **5. Dataset Configuration (`ml-1m.yaml`)**: An example configuration file specifying the data processing and evaluation settings for the MovieLens-1M dataset.

## Setup and Installation

Please follow these steps to set up the environment and run the model.

**Step 1: Clone the Repository**

Clone this repository to your local machine.

```bash
git clone [https://github.com/ydu62203-lang/recbole.git](https://github.com/ydu62203-lang/recbole.git)
cd recbole
```

**Step 2: Create a Virtual Environment (Recommended)**

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

**Step 3: Install Dependencies**

Install the modified RecBole framework and all required packages from the source code.

```bash
pip install -e . --verbose
```

## Running the Experiment

To run the experiment and reproduce the results reported in our paper, use the following command.

**Command:**

```bash
python run_kgat.py --dataset_config=ml-1m
```

This command will start the training and evaluation process for the MKACR model on the MovieLens-1M dataset.

* RecBole will automatically download and process the dataset on the first run.
* The training progress, validation results, and final test results will be displayed in the console.
* Model checkpoints will be saved in the `saved/` directory.

## Cite

If you find our work useful for your research, please consider citing our paper.

```bibtex
@inproceedings{your_lastname_2025_mkacr,
  author    = {First Author and Second Author and You},
  title     = {Multi-Space Knowledge Graph Embedding and Adversarial Contrastive Learning for Recommendation},
  booktitle = {Proceedings of the [Conference Name]},
  year      = {2025}
}
```

This implementation is built on the RecBole library. Please also consider citing their work:

```bibtex
@inproceedings{recbole[1.0],
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}
```
