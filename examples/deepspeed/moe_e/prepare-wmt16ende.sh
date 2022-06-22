#!/bin/bash
# Adapted from "examples/translation/prepare-wmt14en2de.sh"

set -x -e

WORKSPACE="${WORKSPACE?}"
mkdir -p "${WORKSPACE}"
pushd "$WORKSPACE"

echo 'Cloning Moses github repository (for tokenization scripts)...'
[[ ! -d mosesdecoder ]] && git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Installing sentencepiece (for BPE pre-processing)...'
pip install sentencepiece

setup_spm() {
    sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

    git clone https://github.com/google/sentencepiece.git 
    mkdir ./sentencepiece/build
    pushd ./sentencepiece/build
    cmake ..
    make -j $(nproc)
    sudo make install
    sudo ldconfig -v
    popd
}

if [[ "$SETUP_SPM" == 1 && ! -d ./sentencepiece/build ]]; then
    setup_spm
fi

SCRIPTS="${WORKSPACE}/mosesdecoder/scripts"
TOKENIZER="$SCRIPTS/tokenizer/tokenizer.perl"
CLEAN="$SCRIPTS/training/clean-corpus-n.perl"
NORM_PUNC="$SCRIPTS/tokenizer/normalize-punctuation.perl"
REM_NON_PRINT_CHAR="$SCRIPTS/tokenizer/remove-non-printing-char.perl"
# BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz"
    "http://data.statmt.org/wmt16/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v11.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training-parallel-nc-v11/news-commentary-v11.de-en"
)

popd

OUTDIR="${WORKSPACE}/wmt16_en_de"

if [[ ! -d "$SCRIPTS" ]]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
preprocessed="$OUTDIR"
tmp=$preprocessed/tmp
orig="${WORKSPACE}/original"
dev="${WORKSPACE}/dev/newstest2013"

mkdir -p $orig $tmp $preprocessed

pushd $orig

for ((i=0; i < ${#URLS[@]}; ++i)); do
    file="${FILES[i]}"
    if [ -f "$file" ]; then
        echo "$file already exists, skipping download"
    else
        url="${URLS[i]}"
        wget "$url"
        if [ -f "$file" ]; then
            echo "$url successfully downloaded."
        else
            echo "$url FAILED to download."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf "$file"
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf "$file"
        fi
    fi
done

popd

set -e

echo "pre-processing train data..."
for l in $src $tgt; do
    rm "$tmp/train.tags.$lang.tok.$l" || true
    for f in "${CORPORA[@]}"; do
        cat "$orig/$f.$l" \
            >> "$tmp/train.tags.$lang.tok.$l"
            # | perl "$NORM_PUNC" $l \
            # | perl "$REM_NON_PRINT_CHAR" \
            # | perl "$TOKENIZER" -threads 8 -a -l $l >> "$tmp/train.tags.$lang.tok.$l"
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' "$orig/test-full/newstest2014-deen-$t.$l.sgm" \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
    > "$tmp/test.$l"
    # | perl "$TOKENIZER" -threads 8 -a -l $l \
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' "$tmp/train.tags.$lang.tok.$l" > "$tmp/valid.$l"
    awk '{if (NR%100 != 0)  print $0; }' "$tmp/train.tags.$lang.tok.$l" > "$tmp/train.$l"
done

train_pref="$tmp/train"
# train_pref="$orig/"
echo "Learning BPE by sentencepiece on $train_pref.*"
BPE_PREFIX="$preprocessed/sentencepiece.${BPE_TOKENS}k"
[[ ! -s "$BPE_PREFIX.model" ]] && spm_train \
    --input="$train_pref.$src","$train_pref.$tgt" \
    --vocab_size=$BPE_TOKENS --character_coverage=1.0 \
    --model_type=unigram \
    --model_prefix="$BPE_PREFIX"

for L in $src $tgt; do
    for f in {train,valid,test}.$L; do
        INPUT="$tmp/$f"
        OUTPUT="$tmp/bpe.$f"
        echo "Applying BPE to ${f}, output at: ${OUTPUT}"
        spm_encode \
            --model="${BPE_PREFIX}.model" \
            --output_format=piece \
            < "${INPUT}" \
            > "${OUTPUT}"
    done
done

perl "$CLEAN" -ratio 1.5 "$tmp/bpe.train" $src $tgt "$preprocessed/train" 1 250
perl "$CLEAN" -ratio 1.5 "$tmp/bpe.valid" $src $tgt "$preprocessed/valid" 1 250

for L in $src $tgt; do
    cp "$tmp/bpe.test.$L" "$preprocessed/test.$L"
done
