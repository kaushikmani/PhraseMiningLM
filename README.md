# Phrase Mining on Clinical Documents with Language Models

This repository contains the source code and models for our work on a system to easily and efficiently extract quality and meaningful phrases from clinical documents with limited amount of training data. We use deep neural network based language models such as [BERT](https://arxiv.org/abs/1810.04805) and [ELMO](https://arxiv.org/abs/1802.05365) to extract a set of quality phrases.

## 1. Introduction

A vast amount of vital clinical data is available within unstructured texts such as discharge summaries and procedure notes in Electronic Medical Records (EMRs). Automatically transforming such unstructured data into structured units is crucial for effective data analysis in the field of clinical informatics. Recognizing phrases which reveal important medical information in a concise and thorough manner is a fundamental step in this process. We adapt domain-specific deep neural network based language models to effectively and efficiently extract high-quality phrases from clinical documents with limited amount of training data, outperforming the current state-of-the-art techniques in performance and efficiency. In addition, our model trained on the MIMIC-III [Johnson et. al](https://www.nature.com/articles/sdata201635) dataset can be directly applied to a new corpus and it still achieves the best performance among all methods, which shows its great generalizability and can save tremendous labeling effort on the new dataset. The following figure shows architecture of our pipeline. 

![Architecture of our pipeline](/images/architecture.png)

Given a set of clinical documents as input, frequent phrases are extracted from the documents as phrase candidates and features are extracted using a language model for all the phrases to measure the quality of a phrase based on its relevance in the clinical context. These features are then fine-tuned with a classifier, which predicts quality phrases.

## 2. Performance

We compare different language models with open and clinical domain methods such as AutoPhrase, SegPhrase, QuickUMLS and SciSpaCy. Clinical BERT initialized with BioBERT weights and pre-trained on discharge summaries of MIMIC III notes performs the best among all models, however the performance of other BERT based models are comparable. The following table shows performance results of different methods on the MIMIC-III dataset.

![Comparison of performance of different methods on the MIMIC-III dataset](/images/performance.png)

Eventhough open-domain methods outperform our method in terms of efficiency, it is much faster than methods in the clinical domain and gives much better performance. We do not consider the pre-training time of language models, since they can be trained offline and do not need to be trained for every dataset. The following table shows the efficiency results of different
methods on the MIMIC-III dataset.

![Comparison of efficiency of different methods on the MIMIC-III dataset](/images/efficiency.png)


## 3. Experiments

### Requirements

We use the following languages and libraries to run our code.

- python - 3.7.3
- pytorch - 1.0.0
- scikit-learn - 0.20.3
- nltk - 3.3
- numpy - 1.16.4
- pytorch_pretrained_bert - 0.6.2
- transformers - 2.5.1
- allennlp - 0.9.0


## 4. Training your own model for your dataset:

1. Steps for fine-tuning the BERT language model from scratch:

       $ ./phrase_mining.sh
       
 ### Parameters
 
      RAW_TEXT - Raw Text is the input file, where each line is a single document.
      LABELED_FILE - Labeled file is the labeled data for training, where each line is a phrase. 
      TRAIN - Set TRAIN to 1 to train your model on a new dataset, otherwise 0.

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


If you have any questions, please feel free to contact us @ [m.kaushik93@gmail.com](m.kaushik93@gmail.com), [yue.149@buckeyemail.osu.edu](yue.149@buckeyemail.osu.edu) or [sun.397@osu.edu](sun.397@osu.edu]) !

## 4. Citation

@article{,
  title={Phrase Mining on Clinical Documents with Language Models},
  author={Kaushik Mani, Xiang Yue, Bernal Jimenez, Yungui Huang, Simon Lin and Huan Sun},
  journal={AMIA Annual Symposium (Under review)},
  year={2020}
}
