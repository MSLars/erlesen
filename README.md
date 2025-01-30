# ErLeSen

This repository contains the content and data to reproduce the results of the paper 
**Full Synthetic Data for German Text Simplification**. It also provides the main assests of the research peoject
**ErLeSen** which was funded by the german federal ministry.

Its major contributions are:
- framework to create synthetic datasets for text simplification
- source code to train simplification models with created synthetic datasets or other datasets in the corresponding format
- two ai-based evaluation metrics
- code to start inference backend and frontend server to provide a simplification UI

# Setup

## Development

We use [poetry](https://python-poetry.org/) to manage our dependencies. You either need poetry installed on your machine.
Alternatively you can use [conda](https://anaconda.org/) and create a python environment with:

```shell
conda create -n erlesen python=3.12

conda activate erlesen

poetry config virtualenvs.create false --local
```

In the case of a conda installation you should configure poetry to not create a separate virtual environment.
If you use poetry direct on your system, please check the file `poetry.toml` and change the value of the `create`
parameter to `true`.

```toml
[virtualenvs]
create = false
```

Finally install all required dependencies:

```shell
poetry install
```

## Usage

If you just want to use the developed models, either via an API or via the UI, you can create a development environment
and follow the instructions. Otherwise, you can install [docker](https://www.docker.com/) and execute the software in
a containerized environment.

### Dataset Creation

To create a synthetical dataset, first adjust the relevant parameters in `erlesen/synth/create_synthetic_data.py`.
Edit the parameters in the following area:

```python

if __name__ == "__main__":
    logging.info("Starting the script...")

    ################################################################################
    # START Definition of relevant parameter (edit here!)
    ################################################################################
    
    openai_model = "gpt-4o"
    
    # ...
    
    ################################################################################
    # END Definition of relevant parameter (do NOT edit following code!)
    ################################################################################
```

Afterwards execute the python script with (assuming you are in the project root directory) 

```shell
python erlesen/synth/create_synthetic_data.py
```

According to the specified number of training samples `n_samples` the creation takes several hours.
Ensure that `n_test` is smaller than `n_samples`.
You need a valid openai api key as environment variable `OPENAI_API_KEY`. You can create an `.env`
file in the project root directory.

```txt
OPENAI_API_KEY=sk-proj-...
```

The script outputs files `train.jsonl` and `test.jsonl` in `erlesen/data/`

### Training

To finetune a custom model, use the training script `erlesen/training/train_llm.py`.

At first configure the parameters in 

```python
if __name__ == "__main__":
    log("Start training script")

    ################################################################################
    # START Definition of relevant parameter (edit here!)
    ################################################################################
    
    #...

    ################################################################################
    # END Definition of relevant parameter (do NOT edit following code!)
    ################################################################################
```

Afterwards start the training

```shell
python erlesen/training/train_llm.py
```

### Starting th UI

To start the User Interface, fist start a Huggingface TGI inference server.

```shell
bash start_TGI.sh <MODEL_DIR> <MODEL_NAME>
```

The model name can either be one of the provided links to our models on the huggingface hub or a link to a custom model.
If you choose a model on the hub, just past the model name, for example

```shell
bash start_TGI.sh models/ MSLARS/Supermodell
```

If you want to deploy a local model, it has to be in the `models/` folder. You have to specify this folder even if
you use a model from the hub.

After you startet the TGI server, start the ui with

```shell
paython erlesen/ui/app.py
```