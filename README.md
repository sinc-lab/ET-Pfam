# ET-Pfam: Ensemble transfer learning for protein family prediction

This repository contains the data and source code for the manuscript *ET-Pfam: Ensemble transfer learning for protein family prediction*, by S. Escudero, S. A. Duarte, R. Vitale, E. Fenoy, L.A. Bugnon, D.H. Milone and G. Stegmayer, 2025 [preprint](add_link_here). Research Institute for Signals, Systems and Computational Intelligence, [sinc(i)](https://sinc.unl.edu.ar/).

![ET-Pfam](ET-Pfam.png)

ET-Pfam is a novel approach based on ensembles of deep learning classifiers trained with transfer learning to predict functional families in the Pfam database. 

## 1. Environment setup

First, clone the repository and navigate into the project directory:

```
git clone https://github.com/sinc-lab/ET-Pfam
cd ET-Pfam
```

It is recommended to use a Python virtual environment to manage dependencies, such as conda or venv. For example, using conda:

```
conda create -n ET-Pfam python=3.11
conda activate ET-Pfam
```

Once the environment is activated, install the required packages:

```
pip install -r requirements.txt
```

The following 3 sections provide instructions for using a reduced dataset (the *mini dataset*), as it allows quick testing and replication of the main results with significantly lower time and computational costs. The last section summarises the same steps but using the full dataset.

## 2. Test ET-Pfam ensembles

To reproduce the ensemble evaluation presented in the manuscript using pre-trained base models, you will be guided to download the test set embeddings, the trained base models and the ensemble weights (per model and per family).

### 2.1 Download test embeddings

The test embeddings are available in the compressed file [`mini_test.tar.gz`](https://drive.google.com/file/d/1B0T5C8XiVheYTw6IEFAf3LOFsupzPLVs/view?usp=drive_link). You must download it and place it in a local folder called `data/embeddings/` inside the local repository directory `ET-Pfam`. Run the following commands:

```
mkdir -vp data/embeddings/ 
gdown 1B0T5C8XiVheYTw6IEFAf3LOFsupzPLVs --output data/embeddings/mini_test.tar.gz
tar -xzf data/embeddings/mini_test.tar.gz -C data/embeddings/
```

This will create the folder, download the data, and extract the embeddings. Note that the download and extraction may take a few minutes.

### 2.2 Download trained base models and ensemble weights

The file [`mini.tar.gz`](https://drive.google.com/file/d/1SFOHaZy7NtXohyqdT3-LIvH5sC67WyMy/view?usp=drive_link) contains both the trained base models and the ensemble weights. To download the entire folder use:

```
mkdir -vp models/
gdown 1SFOHaZy7NtXohyqdT3-LIvH5sC67WyMy --output models/mini.tar.gz
tar -xzf models/mini.tar.gz -C models/
```

This will create the local folder `models/mini/` with all the required files.

### 2.3 Run ensemble evaluation

Once all the files are in place, you can run the ensemble using all the strategies evaluated in the paper:

```
python3 test_ensemble.py -v all -m models/mini/ -w models/mini/
```

Note that this process, with all strategies evaluations, may take approximately up to 10 hours to complete on a single GPU. The results will be saved to `results/mini/ensemble_metrics.csv`. Also, this folder will contain subdirectories with the ensemble results for each individual strategy.

To quickly inspect the output, you can run the following command:

```
cat results/mini/ensemble_metrics.csv
```

## 3. Train ET-Pfam ensembles

To obtain the ensemble weights for the pre-trained base models downloaded in the previous section, you need to fit the ensemble using the development (dev) partition. Before running the training, you will be guided to download both the trained base models and the test embeddings, provided earlier.  

### 3.1 Download dev embeddings

The dev embeddings are available in a compressed file named [`mini_dev.tar.gz`](https://drive.google.com/file/d/15pHIO7J2awgwkgHR5d4MhK2mt4JC1j54/view?usp=drive_link), located in the provided [Drive folder](https://drive.google.com/drive/folders/1p54V_g4iy-XGjzi0C7LGQP9brb9QnL1k?usp=sharing). 

To download and extract the data, run the following commands from the repository root: 

```
gdown 15pHIO7J2awgwkgHR5d4MhK2mt4JC1j54 --output data/embeddings/mini_dev.tar.gz
tar -xzf data/embeddings/mini_dev.tar.gz -C data/embeddings/
```

### 3.2 Run ensemble training

To train the ensemble weights, run the following command:

```
python3 test_ensemble.py -v all -m models/mini/
```

> [!NOTE]
> This training process may take several hours on a single GPU.

After training, the results will be saved in the folder `results/mini/`. You will find a file named `ensemble_metrics.csv` containing the performance metrics for each strategy, including the Central Window Score (CwS), Sliding Window Area (SwA) and Sliding Window Coverage (SwC).

Additionally, the `results/mini/` folder will contain subfolders named after each strategy, with the outputs from the centered and sliding window tests. The trained ensemble weight files will be stored in the folder `models/mini/`.

## 4. Train base models

### 4.1. Download train embeddings

If you want to train your own base models, you first need to download the training embeddings. These are in the file [`mini_train.tar.gz`](https://drive.google.com/file/d/12-NZPjjeiAKqp4A8gQ9KfIpSzuBawdMp/view?usp=drive_link) inside the provided Drive folder. You can obtain them running the following commands from the repository root:

```
gdown 12-NZPjjeiAKqp4A8gQ9KfIpSzuBawdMp --output data/embeddings/mini_train.tar.gz
tar -xzf data/embeddings/mini_train.tar.gz -C data/embeddings/
```

### 4.2 Run model training

Once all embeddings are available, you can train a base model by running:

```
python3 train_and_test_base.py --output results/mini/<model_name>
```

Replace `<model_name>` with any name you prefer for your experiment.

Inside the specified output directory, you will find the trained model weights saved as `weights.pk`, logs and metrics in the file `train_summary.csv`, a copy of the config file used in `config.json`, and the results for both centered and sliding window tests.

### 4.3. Customize the training configuration

By default, running the training script will train the base models using the settings defined in `config/base.json`. If you want to change hyperparameters, you can edit this JSON file before training.  Inside the json file, you can modify hyperparameters like the window length (`window_len`), learning rate (`lr`) batch size (`batch_size`), and more. After saving your changes, run the training script as shown in the previous step. 

## 5. Reproduce results with full Pfam dataset

You can also reproduce the results from the manuscript using the full Pfam dataset. Note that his process is significantly slower due to the large size of the dataset. Before running any steps for the full dataset, edit the config/base.json and set the `dataset` field to `full`, to indicate that you are working with the full Pfam data instead of the mini version.


### 5.1. Test ET-Pfam ensembles

First, download the required dataset files from the Drive folder and place them into `data/full/` in your local repository. You can download them directly using:

```
gdown --folder 18bD2q-iL2MZE2ddH_-jRp_VDZXoZG8QP --output data/
```

Next, you need to download the test embeddings from the Drive folder located in [`full_test.tar.gz`](https://drive.google.com/file/d/1HFsSCg-z7NqKwdMfxgj8y1-Ux1X7zJnJ/view?usp=drive_link), the trained base models and the ensemble weights for the full dataset. From the repository root, run:

```
gdown 1HFsSCg-z7NqKwdMfxgj8y1-Ux1X7zJnJ --output data/embeddings/full_test.tar.gz
tar -xzf data/embeddings/full_test.tar.gz -C data/embeddings/
gdown 179qTQ-1akdUIkdfjcrxo5X1OOLvEboMl --output models/full.tar.gz
tar -xzf models/full.tar.gz -C models/
```

Once all the files are in place, you can run the ensemble using all voting strategies evaluated in the manuscript:

```
python3 test_ensemble.py -v all -m models/full/ -w models/full/
```

This will reproduce the results for the full Pfam dataset, as reported in the paper.

### 5.2. Train ET-Pfam ensembles

To compute ensemble weights for the full Pfam base models, you need the dev embeddings. Download them from the Drive folder located in [`data/embeddings/full_dev.tar.gz`](https://drive.google.com/file/d/1kRJjLYgzNWwWLHZcZa5oTKiWAEX3cyHn/view?usp=drive_link)  and extract them as follows:

```
gdown 1kRJjLYgzNWwWLHZcZa5oTKiWAEX3cyHn --output data/embeddings/full_dev.tar.gz
tar -xzf data/embeddings/full_dev.tar.gz -C data/embeddings/
```

Then run the ensemble training with:

```
python3 test_ensemble.py -v all -m models/full/
```

### 5.3. Train base models

To train base models from scratch, download the training embeddings from the Drive folder:

```
gdown FILE-ID --output data/embeddings/full_train.tar.gz
tar -xzf data/embeddings/full_train.tar.gz -C data/embeddings/
```

Then run the training script as follows:

```
python3 train_and_test_base.py --output results/full/<model_name>
```

Replace `<model_name>` with any name of your choice for the experiment.
