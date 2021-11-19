# Train the model

OUT_DIR="${OUT_DIR:-'./scratch'}"
DATABIN="${DATABIN?}"

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MKL_THREADING_LAYER=GNU

# RUN_NAME="${1:-}"
# RUN_NAME="moe_g4_ep2_e2_k1_enc"

# FAIRSEQ_SRC="/home/muelnokr/data__/a-munael/code/external/fairseq_master"
# export PYTHONPATH="$FAIRSEQ_SRC:$PYTHONPATH"
USER_DIR=${USER_DIR:-"./user"}
FS_TRAIN="$USER_DIR/train.py"

# ARCH='transformer_vaswani_wmt_en_de_big'
# ARCH='transformer_ds_moe_vaswani_wmt_en_de_big'
# ARCH='transformer_ds_moe_tiny'
# ARCH='transformer_tiny'
ARCH=${ARCH?}

if [[ $ARCH == *ds_moe* ]]; then
    NUM_GPUS=${NUM_GPUS:-8}
    NUM_EXPERTS=${NUM_EXPERTS:-8}
    MOE_MODE=${MOE_MODE:-enc,dec}
    Config=(
        --task 'translation'
        --deepspeed_moe "$MOE_MODE"
            --ep-world-size $NUM_GPUS
            --num-experts   $NUM_EXPERTS
            --top-k 1
        --criterion 'model_and_base'
            --loss-weights '{"base_crit": 1, "experts_gate_loss": 10}'
            --base-criterion 'label_smoothed_cross_entropy'
            --base-criterion-config '{"label_smoothing": 0.1}'
    )
    RUN_NAME_default="moe_g${NUM_GPUS}_ep${NUM_GPUS}_ex${NUM_EXPERTS}_k1_${MOE_MODE//,/}"
else
    Config=(
        --task translation
        --criterion label_smoothed_cross_entropy
            --label-smoothing 0.1
    )
    RUN_NAME_default=baseline
fi

DONT_SAVE="--no-save"
RUN_NAME=${RUN_NAME:-$RUN_NAME_default}

train() {
    SaveDir="${OUT_DIR}/checkpoints/${ARCH}-${RUN_NAME}"
    mkdir -p $SaveDir

    python $FS_TRAIN \
        $DATABIN \
        --seed 43821 \
        --user-dir $USER_DIR \
        --ddp-backend=legacy_ddp --fp16 \
        --arch $ARCH \
        -s 'de' -t 'en' \
        "${Config[@]}" \
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
            --save-dir ${SaveDir} \
            ${DONT_SAVE} \
            --tensorboard-logdir "$OUT_DIR/tb/${ARCH}-${RUN_NAME}"
}

train

