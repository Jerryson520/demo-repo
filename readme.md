# **Microsoft Turing Academic Program**

## **Turing Latural Language Representation Model (TNLR-v5) Overview**
We are excited to release a private preview of the Turing Natural Language Representation v5 (TNLRv5) model to our MS-TAP partners as part of our commitment to responsible AI development. MS-TAP partners will have access to the base (12-layer, 768 hidden, 12 attention heads, 184M parameters) and large (24-layer, 1024 hidden, 16 attention heads, 434M parameters) T-NLRv5 model.   

T-NLRv5 integrates some of the best modeling techniques developed by Microsoft Research, Azure AI, and Microsoft Turing. The models are pretrained at large scale using an efficient training framework based on FastPT and DeepSpeed. T-NLRv5 is the state of the art at the top of SuperGLUE and GLUE leaderboards, further surpassing human performance and other models. Notably, T-NLRv5 first achieved human parity on MNLI and RTE on the GLUE benchmark, the last two GLUE tasks which human parity had not yet met. In addition, T-NLRv5 is more efficient than recent pretraining models, achieving comparable effectiveness with 50% fewer parameters and pretraining computing costs. 

T-NLRv5 is largely based on our recent work, [COCO-LM](https://arxiv.org/abs/2102.08473), a natural evolution of pretraining paradigm converging the benefits of ELECTRA-style models and corrective language model pretraining. Read more about TNLRv5 in our [blog post](https://www.microsoft.com/en-us/research/blog/efficiently-and-effectively-scaling-up-language-model-pretraining-for-best-language-representation-model-on-glue-and-superglue/). 

## **Model Setup**
1. Install *git lfs*:
   

2. *Clone the repository* using personal token:  
   * Personal token:
      Create a personal token following [this](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) link.  

   * Clone the repository
      ```bash
      git lfs clone https://username@github.com/username/mstap-TNLR-harvard-cai-lu
      ```
      [Password: Personal token]
   
3. *Pytorch* >= 1.6, CUDA version
   * Installing PyTorch following [this](https://pytorch.org/get-started/previous-versions/) link


   * Apex will be installed successfully if:
      ```bash
      runtime api version: nvcc -V
      ```
      ```python
      import torch
      print(torch.__version__)
      ```
      are the same.

4. *Apex* (same version as CUDA PyTorch)
   Install apex following instruction [here](https://github.com/NVIDIA/apex).


5. *Transformers*
   Install transformers using:
   ```bash
   pip install transformers==2.10.0 
   ```    

## **GLUE Finetuning**  
### Downloading the GLUE dataset
The [GLUE dataset](https://gluebenchmark.com/tasks) can be downloaded by running the following script

```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
python src/download_glue_data.py
```

### MNLI (base)
```shell
 # Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_base_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-base-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_base_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_base_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run
export BSZ=32
export LR=1e-5
export EPOCH=5
export WD=0.1
export WM=0.0625

CUDA_VISIBLE_DEVICES=0 python src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```

#### Sample results (Acc):

`--seed`: 1

```
MNLI-m: 91.726
MNLI-mm: 91.456
```


### MNLI (large)
#### 1. Single GPU
```shell
 # Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_large_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-large-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-large-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run.
export BSZ=32
export LR=3e-6
export EPOCH=2
export WD=0.1
export WM=0.0625
CUDA_VISIBLE_DEVICES=0 python src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```
 
 #### Sample results (Acc):

`--seed`: 1

```
MNLI-m: 91.726
MNLI-mm: 91.456
```


#### 2. Multiple GPUs
```shell
 # Set path to read training/dev dataset that was downloaded in the previous step
export DATASET_PATH=${HOME_DIR}/glue_data/MNLI

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/mnli_large_ft/

export TASK_NAME=mnli

# Set model name (or checkpoint path) for finetuning 
export CKPT_PATH=tnlrv5-large-cased

# Set max sequence length
export MAX_LEN=512

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-large-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.tnlrv5_large_cased.$MAX_LEN.cache

# Setting the hyperparameters for the run.
# per_gpu_train_batch_size = train_batch_size / num_gpus = 32 / 8 = 4
export BSZ=4
export LR=3e-6
export EPOCH=2
export WD=0.1
export WM=0.0625
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 src/run_classifier.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH --task_name $TASK_NAME \
    --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --do_train --evaluate_during_training --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
    --max_seq_length $MAX_LEN --per_gpu_train_batch_size $BSZ --learning_rate $LR \
    --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --dropout_prob 0.1 --cls_dropout_prob 0.1 \
    --fp16 --fp16_opt_level O2 --seed 1
 ```

 #### Sample results (Acc):

`--seed`: 1

```
MNLI-m: 91.726
MNLI-mm: 91.456
```



## **SQuAD 2.0 Fine-tuning (Base & Large)**
[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 


### 1. Single GPU
 ```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
# Set path to the location where the data will be downloaded
export DATASET_PATH=${HOME_DIR}/squad_data/

# Download the train & dev datset
mkdir -p ${DATASET_PATH}
# Train datset
export TRAIN_FILE=${DATASET_PATH}/train-v2.0.json
wget -O $TRAIN_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# Dev datset
export DEV_FILE=${DATASET_PATH}/dev-v2.0.json
wget -O $DEV_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/squad_ft/

# Set path to the model checkpoint you need to test 
export CKPT_PATH=tnlrv5-base-cased

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${TRAIN_FILE}_tnlrv5_base_cased.384doc.cache
export DEV_CACHE=${DEV_FILE}_tnlrv5_base_cased.384doc.cache

# Setting the hyperparameters for the run.
export BSZ=32
export LR=3e-5
export EPOCH=3
CUDA_VISIBLE_DEVICES=0 python src/run_squad.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --train_file $TRAIN_FILE --predict_file $DEV_FILE \
    --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --do_train --do_eval \
    --per_gpu_train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH --gradient_accumulation_steps 1 \
    --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
    --version_2_with_negative --seed 1 --max_grad_norm 0 \
    --weight_decay 0.1 --warmup_ratio 0.0625 \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --fp16_opt_level O2 --fp16
```


### 2. Multiple GPUs
```shell
# Set path for this repository
export HOME_DIR=~/mstap-TNLR-harvard-cai-lu
cd ${HOME_DIR}
# Set path to the location where the data will be downloaded
export DATASET_PATH=${HOME_DIR}/squad_data/

# Download the train & dev datset
mkdir -p ${DATASET_PATH}
# Train datset
export TRAIN_FILE=${DATASET_PATH}/train-v2.0.json
wget -O $TRAIN_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# Dev datset
export DEV_FILE=${DATASET_PATH}/dev-v2.0.json
wget -O $DEV_FILE https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# Set path to save the finetuned model and result score
export OUTPUT_PATH=${HOME_DIR}/squad_ft/

# Set path to the model checkpoint you need to test 
export CKPT_PATH=tnlrv5-base-cased

# Set config file
export CONFIG_FILE=${HOME_DIR}/configs/tnlrv5-base-cased.json

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${TRAIN_FILE}_tnlrv5_base_cased.384doc.cache
export DEV_CACHE=${DEV_FILE}_tnlrv5_base_cased.384doc.cache

# Setting the hyperparameters for the run.
# per_gpu_train_batch_size = train_batch_size / num_gpus = 32 / 8 = 4
export BSZ=4
export LR=3e-5
export EPOCH=3
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 src/run_squad.py \
    --model_type tnlrv5 --model_name_or_path $CKPT_PATH \
    --config_name $CONFIG_FILE --tokenizer_name tnlrv5-cased \
    --train_file $TRAIN_FILE --predict_file $DEV_FILE \
    --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
    --do_train --do_eval \
    --per_gpu_train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH --gradient_accumulation_steps 1 \
    --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
    --version_2_with_negative --seed 1 --max_grad_norm 0 \
    --weight_decay 0.1 --warmup_ratio 0.0625 \
    --fp16_init_loss_scale 128.0 --adam_epsilon 1e-6 --adam_betas "0.9,0.98" \
    --fp16_opt_level O2 --fp16
 ```


## Papers
* [COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining](https://arxiv.org/abs/2102.08473)
* [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
* [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
## Contributing 
See [CONTRIBUTING.md](./CONTRIBUTING.md). 

## License 
See [LICENSE.txt](./LICENSE.txt). 
## Security 
See [SECURITY.md](./SECURITY.md). 

## Support 
Please email us at turing-academic@microsoft.com for troubleshooting, or file an issue through the repo 



