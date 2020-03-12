#!/bin/bash
# RAW_TEXT is the input file for Phrase Mining, where each line is a single document.
RAW_TEXT='data/input.txt'
# LABELED_FILE is the name and location of the labeled data file used by our model.
LABELED_FILE='data/training_set.txt'
# A threshold of raw frequency is specified for frequent phrase mining, which will generate a candidate set of phrases.
MIN_FREQ=10
# A threshold of length is specified for frequent phrase mining, which will generate a candidate set of phrases.
MAX_LENGTH=6
# Set to 1 if you want to use GPU to run our model.
CUDA=1
# PRETRAINED_WEIGHTS is the name and location of the pretrained_weights for BERT model.
PRETRAINED_WEIGHTS='model/pretrained_bert_tf/biobert_pretrain_output_disch_100000'
# OUTPUT_MODEL is the name and location of the BERT model used for phrase mining.
OUTPUT_MODEL='model/clinical_bert_best_model.bin'
# OUTPUT_CONFIG is the name and location of the BERT config used for phrase mining.
OUTPUT_CONFIG='model/clinical_bert_best_model_config.bin'
# OUTPUT_VOCAB is the name and location of the BERT vocab used for phrase mining.
OUTPUT_VOCAB='model/clinical_bert_best_model_vocab.bin'
# BERT DIMENSIONS is the dimension size of the BERT Model
BERT_DIMENSIONS=768
# LR is the learning rate of the fine-tuning model.
LR=2e-5
# EPOCHS is the number of epochs to fine-tune the pretrained BERT model
EPOCHS=4
# QUALITY_OUTPUT is the output of our model, where each line is a quality phrase
QUALITY_OUTPUT='output/quality_phrases.txt'
# Set to 1 if you want to extract frequent N grams.
FREQUENT_NGRAM=0
# Set to 1 if you want to fine-tune a pretrained model
TRAIN=1

#Frequent Phrase Generation
if [ ${FREQUENT_NGRAM} -eq 1 ]; then
python src/frequent_ngram_gen.py \
    --input_file $RAW_TEXT \
    --length $MAX_LENGTH \
    --frequency $MIN_FREQ
fi

if [ ${CUDA} -eq 1 ]; then
    if [ ${TRAIN} -eq 1 ]; then
        python src/main.py \
            --input_file $RAW_TEXT \
            --output_file $QUALITY_OUTPUT \
            --training_file $LABELED_FILE \
            --num_epochs $EPOCHS \
            --pretrained_weights $PRETRAINED_WEIGHTS \
            --output_model $OUTPUT_MODEL \
            --output_config $OUTPUT_CONFIG \
            --output_vocab $OUTPUT_VOCAB \
            --cuda \
            --train
    else
        python src/main.py \
            --input_file $RAW_TEXT \
            --output_file $QUALITY_OUTPUT \
            --num_epochs $EPOCHS \
            --pretrained_weights $PRETRAINED_WEIGHTS \
            --output_model $OUTPUT_MODEL \
            --output_config $OUTPUT_CONFIG \
            --output_vocab $OUTPUT_VOCAB \
            --cuda
    fi
else
    if [ ${TRAIN} -eq 1 ]; then
        python src/main.py \
            --input_file $RAW_TEXT \
            --output_file $QUALITY_OUTPUT \
            --training_file $LABELED_FILE \
            --num_epochs $EPOCHS \
            --pretrained_weights $PRETRAINED_WEIGHTS \
            --output_model $OUTPUT_MODEL \
            --output_config $OUTPUT_CONFIG \
            --output_vocab $OUTPUT_VOCAB \
            --train
    else
         python src/main.py \
            --input_file $RAW_TEXT \
            --output_file $QUALITY_OUTPUT \
            --num_epochs $EPOCHS \
            --pretrained_weights $PRETRAINED_WEIGHTS \
            --output_model $OUTPUT_MODEL \
            --output_config $OUTPUT_CONFIG \
            --output_vocab $OUTPUT_VOCAB
    fi
fi
