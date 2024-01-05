[![Issues](https://img.shields.io/github/issues/webis-de/precision-argument-retrieval?style=flat-square)](https://github.com/webis-de/precision-argument-retrieval/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/precision-argument-retrieval?style=flat-square)](https://github.com/webis-de/precision-argument-retrieval/commits)
[![License](https://img.shields.io/github/license/webis-de/precision-argument-retrieval?style=flat-square)](LICENSE)

# 🆚 precision-argument-retrieval

Code and data for the paper "DeepCT-enhanced Lexical Argument Retrieval".

## Installation

1. Install [Python 3.9](https://python.org/downloads/)
2. Create and activate virtual environment:
    ```shell
    python3.9 -m venv venv/
    source venv/bin/activate
    ```
3. Install dependencies:
    ```shell
    pip install -e .
    ```

## Usage
Follow these steps to prepare DeepCT and train it on a Slurm cluster.

### Prepare DeepCT Image in Slurm

First, create your enroot image:
```shell
srun -c 4  --mem=100G --container-image=nvcr.io#nvidia/tensorflow:20.06-tf1-py3 --container-name=DeepCT --pty echo "Image created sucessfully"
```

Clone DeepCT:
```shell
srun -c 4  --mem=100G --container-name=DeepCT --container-writable --pty git clone https://github.com/AdeDZY/DeepCT.git
```

Download BERT:
```shell
srun -c 4  --mem=100G --container-name=DeepCT --container-writable --pty bash -c 'wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip && mkdir bert-uncased_L-12_H-768_A-12 && unzip uncased_L-12_H-768_A-12.zip -d bert-uncased_L-12_H-768_A-12 && rm -f uncased_L-12_H-768_A-12.zip' 
```

### Train DeepCT with Slurm

After you have executed the steps to [prepare the DeepCT image in Slurm](#prepare-deepct-image-in-slurm), you can train the model with the following commands.

Assume you have the following files/directories:
- `<PATH-TO-TRAIN-DOCTERM-RECALL>`: the training data file.
- `<OUTPUT-DIRECTORY>`: directory where the trained model will be stored.

You can train your model with the following command:

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

### Train DeepCT with Slurm in case `--chdir` does not work:

First run following command in order to get in interactive Bash-Shell in Container
```shell
srun \
    --gres gpu:ampere:1 -c 4  --mem=100G \
    --container-mounts=<PATH-TO-TRAIN-DOCTERM-RECALL>,<OUTPUT-DIRECTORY> \
    --container-name=DeepCT --pty \
    bash
```

Then `cd` to the downloaded Git repository `cd DeepCT` and run the rest of the original script:
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

## License

The source code in this repository is licensed under the [MIT License](LICENSE).
