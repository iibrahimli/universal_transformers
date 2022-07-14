# Universal Transformer

Authors:
- Caterina Roncalli (caterina.roncalli@studium.uni-hamburg.de)
- Imran Ibrahimli (imran.ibrahimli@studium.uni-hamburg.de)

Reference paper:
 - https://openreview.net/forum?id=HyzdRiR9Y7


## Setup

First, clone this repository:
    
```bash
git clone https://github.com/iibrahimli/universal_transformers.git
```

Then, create a virtual environment and install the dependencies in `requirements.txt`:

```bash
# change to the directory of the repository
cd universal_transformers

# create venv named ut_venv
python3 -m venv ut_venv

# activate venv
source ut_venv/bin/activate

# update pip & install dependencies
pip install -U pip
pip install -r requirements.txt
```

## Training

To train the model, we strongly advise using multiple GPUs.

### Algorithmic tasks

The algorithmic tasks are randomly generated.

```
python train_algorithmic.py
```

### WMT-14 EN-DE

Training the model on WMT-14 EN-DE dataset is implemented using PyTorch Distributed and therefore is run via `torchrun`. The training script is `train_wmt14.py`.

```
torchrun --standalone \
         --nnodes=1 \
         --nproc_per_node=<N_GPUS>\
         train_wmt14.py \
         --batch_size 32 \
         --max_seq_len 100 \
         --lr_mul 0.2
```

where <N_GPUS> is the number of GPUs you want to use.