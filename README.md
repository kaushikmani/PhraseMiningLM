# ClinPhrase: Extracting Quality Phrases from Clinical Documents

## 1. Introduction

This repository contains the source code and models for our paper, a system to easily and efficiently extract quality and meaningful phrases from clinical documents with limited amount of training data. We use deep neural network based language models such as BERT to extract a set of quality phrases.

## 2. Requirements
- python - 3.7.3
- pytorch - 1.0.0
- scikit-learn - 0.20.3
- nltk - 3.3
- numpy - 1.16.4
- pytorch_pretrained_bert - 0.6.2
- transformers - 2.5.1
- allennlp - 0.9.0

 
### Parameters

        RAW_TEXT - Raw Text is the input for our model, where each line is a single document.
        TRAIN - Set TRAIN to 0 to extract phrases based on a model fine-tuned on another dataset.
 
Other parameters are important for fine-tuning your own model and are explained below.

## 4. Training your own model for your dataset:

1. Steps for fine-tuning the BERT language model from scratch:

       $ ./phrase_mining.sh
       
 ### Parameters
 
      RAW_TEXT - Raw Text is the input file, where each line is a single document.
      LABELED_FILE - Labeled file is the labeled data for training, where each line is a phrase. 

In the labeled file, the phrases and labels must be seperated by a tab(\t) and should be in lowercase. We recommend running the file frequent_ngram_gen.py to extract the frequent N-grams in the input file. This can be done using:
  
      $ python frequent_ngram_gen.py --input_file data/input.txt --frequency 10 --length 6 

Here 'data/input.txt' is the name of the input file, frequency is the minimum threshold frequency(default value - 10) and length is the maximum length threshold(default value - 6) for N-gram extraction. The frequent N-grams are present in the file `tmp/frequent_multi_word_phrases.txt`.

Format of the labeled data file should be: <br />
&nbsp;&nbsp;&nbsp;&nbsp;lung cancer  good <br />
&nbsp;&nbsp;&nbsp;&nbsp;watching television  bad <br />
&nbsp;&nbsp;&nbsp;&nbsp;brain tumour  good <br />
&nbsp;&nbsp;&nbsp;&nbsp;morning sickness  good <br />
&nbsp;&nbsp;&nbsp;&nbsp;this man bad <br />
    
    MIN_FREQ - A threshold of raw frequency is specified for frequent phrase mining, which will generate a candidate set of phrases.
    MAX_LENGTH - A threshold of length is specified for frequent phrase mining, which will generate a candidate set of phrases.
    CUDA - Set to 1 if you want to use GPU to run ClinPhrase. We recommend using a GPU for faster results, since we use deep neural network models in our system.
    PRETRAINED_WEIGHTS - The name and location of the pre-trained weights for BERT Model.
    OUTPUT_MODEL - The name and location of the BERT Model. 
    OUTPUT_CONFIG - The name and location of the config file of BERT Model. 
    OUTPUT_VOCAB - The name and location of the vocab file of BERT Model.
    BERT_DIMENSIONS - Dimension size of the BERT Model.
    LR - The learning rate for fine-tuning the BERT Model.
    EPOCHS - The number of epochs for fine-tuning the BERT Model.  
    QUALITY_OUTPUT - The name and location of the output file. In the output of our model, each line is a quality phrase.
    FREQUENT_NGRAM - Set to 0 if you have already run frequent_ngram_gen script mentioned previously for the same input file.
        

We also provide a fine-tuned ClinicalBERT by [Alsentzer et al., 2019](https://www.aclweb.org/anthology/W19-1909/) model for phrase mining task fine-tuned on the MIMIC-III dataset provided by [Johnson et al., 2016](https://www.nature.com/articles/sdata201635). The pre-trained ClinicalBERT model by Alsentzer et al. is available [here](https://github.com/EmilyAlsentzer/clinicalBERT). Our fine-tuned model for Phrase Mining is available [here](https://drive.google.com/open?id=1P3NnxjaHTLa40aE9dt4gSA9y4xYmcuRg).

We have also provided scripts to run ELMo(eval_elmo.py) and BERT(eval_bert.py) models on a simple training and testing set. 


If you have any questions, please feel free to contact us!

## 4. Citation

Please contact the authors for citation.
