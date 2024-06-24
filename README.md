[![Issues](https://img.shields.io/github/issues/webis-de/argmining24-deepct-lexical-argument-retrieval?style=flat-square)](https://github.com/webis-de/argmining24-deepct-lexical-argument-retrieval/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/argmining24-deepct-lexical-argument-retrieval?style=flat-square)](https://github.com/webis-de/argmining24-deepct-lexical-argument-retrieval/commits)
[![License](https://img.shields.io/github/license/webis-de/argmining24-deepct-lexical-argument-retrieval?style=flat-square)](LICENSE)

# 🆚 argmining24-deepct-lexical-argument-retrieval

Code and data for the paper "DeepCT-enhanced Lexical Argument Retrieval".

## Installation

1. Install [Python 3.9](https://python.org/downloads/) or higher.
2. Create and activate the virtual environment:

    ```shell
    python3.9 -m venv venv/
    source venv/bin/activate
    ```

3. Install dependencies:

    ```shell
    pip install -e .
    ```

## Usage

Follow these steps to prepare [DeepCT](https://github.com/AdeDZY/DeepCT) and train it on a Slurm cluster.

### Prepare DeepCT Image in Slurm

First, create an [enroot](https://github.com/NVIDIA/enroot) image:

```shell
srun -c 4  --mem=100G --container-image=nvcr.io#nvidia/tensorflow:20.06-tf1-py3 --container-name=DeepCT --pty echo "Image created sucessfully"
```

In this image and using Slurm, clone the [DeepCT repository](https://github.com/AdeDZY/DeepCT):

```shell
srun -c 4  --mem=100G --container-name=DeepCT --container-writable --pty git clone https://github.com/AdeDZY/DeepCT.git
```

Download BERT:

```shell
srun -c 4  --mem=100G --container-name=DeepCT --container-writable --pty bash -c 'wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip && mkdir bert-uncased_L-12_H-768_A-12 && unzip uncased_L-12_H-768_A-12.zip -d bert-uncased_L-12_H-768_A-12 && rm -f uncased_L-12_H-768_A-12.zip' 
```

### Train DeepCT with Slurm

After executing the steps to [prepare the DeepCT image in Slurm](#prepare-deepct-image-in-slurm), train the model with the following commands.

This guide assumes that you have the following files/directories:

- `<PATH-TO-TRAIN-DOCTERM-RECALL>`: the training data file.
- `<OUTPUT-DIRECTORY>`: directory where the trained model will be stored.

Train the model with the following command:

```shell
srun \
    --gres gpu:ampere:1 -c 4  --mem=100G \
    --container-mounts=<PATH-TO-TRAIN-DOCTERM-RECALL>,<OUTPUT-DIRECTORY> \
    --chdir /workspace/DeepCT \
    --container-name=DeepCT --pty \
    python3 run_deepct.py \
        --task_name=marcodoc \
        --do_train=true \
        --do_eval=false \
        --do_predict=false \
        --data_dir=<PATH-TO-TRAIN-DOCTERM-RECALL> \
        --vocab_file=/workspace/bert-uncased_L-12_H-768_A-12/vocab.txt \
        --bert_config_file=/workspace/bert-uncased_L-12_H-768_A-12/bert_config.json \
        --init_checkpoint=/workspace/bert-uncased_L-12_H-768_A-12/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=16 \
        --learning_rate=2e-5 \
        --num_train_epochs=3.0 \
        --recall_field=title \
        --output_dir=<OUTPUT-DIRECTORY>
```

### Train DeepCT with Slurm (in case `--chdir` does not work)

First, run the following command in order to start an interactive shell in the container:

```shell
srun \
    --gres gpu:ampere:1 -c 4  --mem=100G \
    --container-mounts=<PATH-TO-TRAIN-DOCTERM-RECALL>,<OUTPUT-DIRECTORY> \
    --container-name=DeepCT --pty \
    bash
```

Then `cd` to the downloaded Git repository and run the rest of the original script:

```shell
python3 run_deepct.py \
    --task_name=marcodoc \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --data_dir=<PATH-TO-TRAIN-DOCTERM-RECALL> \
    --vocab_file=/workspace/bert-uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/workspace/bert-uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/workspace/bert-uncased_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=16 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --recall_field=title \
    --output_dir=<OUTPUT-DIRECTORY>
```

## Download Touché runs

These commands will download the runs from the
[Touché 2020 task 1](https://touche.webis.de/clef20/touche20-web/argument-retrieval-for-controversial-questions).

```shell
wget -c https://zenodo.org/records/6873564/files/touche2020-task1-runs-args-me-corpus-version-2020-04-01.zip
unzip touche2020-task1-runs-args-me-corpus-version-2020-04-01.zip
rm touche2020-task1-runs-args-me-corpus-version-2020-04-01.zip
mkdir -p data/runs_touche/
mv touche2020-task1-runs-args-me-corpus-version-2020-04-01/ data/runs_touche/2020/
```

These commands will download the runs from the
[Touché 2021 task 1](https://touche.webis.de/clef21/touche21-web/argument-retrieval-for-controversial-questions).

```shell
wget -c https://zenodo.org/records/6873566/files/touche2021-task1-runs.zip
unzip touche2021-task1-runs.zip
rm touche2021-task1-runs.zip
mkdir -p data/runs_touche/
mv touche2021-task1-runs/ data/runs_touche/2021/
```

## License

The source code in this repository is licensed under the [MIT License](LICENSE).
