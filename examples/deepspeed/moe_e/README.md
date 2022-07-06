# Using [deepspeed](https://github.com/microsoft/DeepSpeed) in Fairseq

This page includes instructions for an example implementation of an MoE Transformer model based on the deepspeed package.

## Resources

### Requirements
Consider working in a separate python virtual env (e.g. via `conda create --name myclone --clone myenv`).

```bash
# Install fairseq from this source
pip install --editable ./
python setup.py build_ext --inplace
pip install deepspeed
pip install sacremoses
# The fairseq commit used here is not the latest, and is incompatible with sacrebleu>=2.*
pip install 'sacrebleu==1.5.*'
```

### Datasets
- WMT14 English-French: https://www.statmt.org/wmt14/translation-task.html#Download
- WMT16 English-German: https://www.statmt.org/wmt16/translation-task.html#download

<!-- ###  -->


## Training a new model on WMT'16 En-De

First, specify a workspace directory where we'll store the data, models, and any other artifacts. E.g.

```sh
export WORKSPACE="./workspace"
```

The steps are:
1. Download the raw text data in `<lang1>`, `<lang2>` file pairs. This's their original format from WMT.
2. Extract the corresponding raw text file pairs.
3. If needed, clean and tokenize the raw texts.
4. Train a subword vocabulary (BPE) using, for example, [sentencepiece](https://github.com/google/sentencepiece).
5. Encode the texts using the trained vocabulary.
6. Binarize the dataset using `fairseq-preprocess`.
7. Train using `fairseq-train`, pointing to the user extensions directory `./user` next to this README.

### Expected results
| Metric                | Dense    | Deepspeed MoE x8 |
|-----------------------|----------|------------------|
| Train ntokens/batch   | 9.47E+05 | 1.01E+06         |
| Train tokens/sec      | 138,447  | 70,964           |
| Val tokens/sec        | 50,438   | 7,000            |
| Best valid BLEU       | 36.6     | 36.13            |
| Steps to Best         | 18,756   | 5,740            |
| Epochs to Best        | 156      | 51               |
| Comparable Valid BLEU | 36.12    | 36.13            |
| Steps to Comparable   | 9,600    | 5,740            |
| N Experts             | 1        | 8                |
| seconds/update        | 6.67     | 14.2             |


### 1-5. Prepare the texts
```bash
bash ./prepare-wmt16ende.sh
```

### 6. Preprocess the dataset with a joined dictionary
```bash
TEXT="${WORKSPACE?}/wmt16_en_de"
DATABIN="${WORKSPACE}/wmt16_en_de/databin"
mkdir -p "$DATABIN"
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref "$TEXT/train" \
    --validpref "$TEXT/valid" \
    --testpref  "$TEXT/test" \
    --destdir   "${DATABIN}" \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

### 7. Train a model
> See also: [run.sh](./run.sh)

1. The dir at the `$OUT_DIR` env variable will include training artifacts like checkpoints (`/checkpoints`) and tensorboard logs (`/tb`).
2. Specify an architecture. Architectures based on the vanilla Transformer (`transformer_*`) have been adapted in the user directory as `transformer_ds_moe_*`.
   1. Example: `transformer_tiny` ==> `transformer_ds_moe_tiny`.
   2. `export ARCH=transformer_ds_moe_{}` before executing `run.sh`.

Use any recipe for a Transformer model from fairseq and add the following arguments (assigned to the array variable `${Config[@]}`):
```bash
    NUM_GPUS=${NUM_GPUS:-8}
    NUM_EXPERTS=${NUM_EXPERTS:-8}
    MOE_MODE=${MOE_MODE:-enc,dec}
    Config=(
        --ddp-backend=legacy_ddp
        --task 'translation_deepspeed'
        --deepspeed_moe "$MOE_MODE"
            --ep-world-size $NUM_GPUS
            --num-experts   $NUM_EXPERTS
            --top-k 1
        --criterion 'model_and_base'
            --loss-weights '{"base_crit": 1, "experts_gate_loss": 10}'
            --base-criterion 'label_smoothed_cross_entropy'
            --base-criterion-config '{"label_smoothing": 0.1}'
    )
    RUN_NAME="moe_g${NUM_GPUS}_ep${NUM_GPUS}_ex${NUM_EXPERTS}_k1_${MOE_MODE//,/}"
```

#### ***Config details***
| Argument                  | Values                                                                       | Effect                                                                                                                                                                            |
|---------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--ddp-backend`           | `legacy_ddp`                                                                 | This code has been tested only with legacy_ddp mode                                                                                                                               |
| `--task`                  | `translation_deepspeed`                                                      | No functional changes from the base `translation`                                                                                                                                 |
| `--deepspeed_moe`         | `enc`, `dec`, `enc,dec`                                                      | Enables MoE layers in `enc`oder, `dec`oder or both                                                                                                                                |
| `--ep-world-size`         | `$NUM_GPUS`                                                                  | Expert world size must equal # of GPUs for now                                                                                                                                    |
| `--num-experts`           | Number divisible by `$NUM_GPUS`                                              | Total number of experts in the model.                                                                                                                                             |
| `--top-k`                 | `1`, `2`                                                                     | Expert selection mode.                                                                                                                                                            |
| `--criterion`             | `model_and_base`                                                             | The `model_and_base` custom criterion uses any loss returned by the model (e.g. `expert_gate_loss` in this case) plus another specified "base criterion".                         |
| `--loss-weights`          | String of a json dictionary of `"{crit_name}": float(crit_weight)`  KV pairs | The final loss is the weighted sum of the model loss(es) and the base criterion's loss, weighted by the provided value or `1.0` otherwise. `null` ignores the corresponding loss. |
| `--base-criterion`        | A fairseq criterion's CLI name                                               | The base criterion for the main objective. E.g. `label_smoothed_cross_entropy`.                                                                                                   |
| `--base-criterion-config` | String of a json dictionary of the arguments of init for the base criterion  | -                                                                                                                                                                                 |

#### ***Example train command***
```bash
    OUT_DIR="${WORKSPACE:-"."}/train_artifacts"
    SaveDir="${OUT_DIR?}/checkpoints/${ARCH}-${RUN_NAME}"

    # ARCH='transformer_ds_moe_vaswani_wmt_en_de_big'
    ARCH='transformer_ds_moe_tiny'

    python "${USER_DIR?}/train.py" \
        "${DATABIN?}" \
        --seed 43821 \
        --user-dir "$USER_DIR" \
                                    "${Config[@]}" \
        --fp16 \
        --arch $ARCH \
        -s 'de' -t 'en' \
        --reset-optimizer \
        --optimizer adam \
            --adam-betas '(0.9, 0.98)' \
            --clip-norm 0.0 \
        --lr 5e-4 \
            --dropout 0.3 \
            --weight-decay 0.0001 \
            --lr-scheduler inverse_sqrt \
            --warmup-updates 4000 \
        --max-update 300000 \
        --max-tokens-valid "${MAX_TOKENS:-8192}" \
        --max-tokens "${MAX_TOKENS:-8192}" \
            --update-freq "${UPDATE_FREQ:-16}" \
        --validate-interval-updates 20 \
        --eval-bleu \
            --scoring sacrebleu \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-detok moses \
            --eval-bleu-remove-bpe \
            --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 1 \
            --save-interval-updates 100 \
            --keep-interval-updates 1 \
            --save-dir "${SaveDir}" \
            --tensorboard-logdir "${OUT_DIR?}/tb/${ARCH}-${RUN_NAME}"
```

Note that the `--fp16` flag requires CUDA 9.1 or greater and a Volta GPU or newer.

***IMPORTANT:*** You will get better performance by training with big batches and
increasing the learning rate. If you want to train the above model with big batches
(assuming your machine has 8 GPUs):
- add `--update-freq 16` to simulate training on `8x16=128` GPUs
- increase the learning rate; to e.g. `0.001`

## Evaluate

> ***Not ready.***

Now we can evaluate our trained model.

Next, generate translations using a beam width of 4 and length penalty of 0.6:
```bash
python "${USER_DIR?}/generate.py" \
    "${DATABIN?}" \
    --seed 43821 \
    --user-dir "$USER_DIR" \
    --path "${LatestCheckpoint}" \
    --max-tokens-valid "${MAX_TOKENS:-8192}" \
    --save-dir "${SaveDir}" \
    --tensorboard-logdir "${OUT_DIR?}/tb/${ARCH}-${RUN_NAME}" \
    --beam 4 --lenpen 0.6 --remove-bpe > gen.out
```

To compute detokenized BLEU with sacrebleu (preferred):
```bash
bash scripts/sacrebleu.sh wmt14/full en de gen.out
```
<!-- # BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3 = 28.6 59.3/34.3/22.1/14.9 (BP = 1.000 ratio = 1.016 hyp_len = 63666 ref_len = 62688) -->
