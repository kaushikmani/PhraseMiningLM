import argparse
import torch
import random
import numpy as np
from tqdm import trange
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from transformers import *
import time
import os
import sys
from collections import OrderedDict
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
label_encoder = LabelEncoder()
min_max_scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder()

curr_dir = os.getcwd()


def parse_args():

    parser = argparse.ArgumentParser(description='Clinical Phrase Extraction')
    parser.add_argument('--input_file', type=str, default=curr_dir + '/data/input.txt', help='Name and Location of the text corpus')
    parser.add_argument('--output_file', type=str, default=curr_dir + '/data/quality_output.txt', help='Name and Location of the extracted phrases')
    parser.add_argument('--length', type=int, default=6, help='Maximum length of N-Grams extracted')
    parser.add_argument('--frequency', type=int, default=10, help='Minimum frequency threshold of N-Grams')
    parser.add_argument('--train', action='store_true', help='Train Classifier')
    parser.add_argument('--training_file', type=str, default=curr_dir + '/data/training_set.txt', help='Name and Location of annotated phrases')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--pretrained_weights', type=str, default=curr_dir + '/model/pretrained_bert_tf/biobert_pretrain_output_disch_100000', help='Pretrained weights for BERT model')
    parser.add_argument('--output_model', type=str, default=curr_dir + '/model/clinical_bert_best_model.bin', help='Name and Location of bert model')
    parser.add_argument('--output_config', type=str, default=curr_dir + '/model/clinical_bert_best_model_config.bin', help='Name and Location of bert model')
    parser.add_argument('--output_vocab', type=str, default=curr_dir + '/model/clinical_bert_best_model_vocab.bin', help='Name and Location of bert model')
    parser.add_argument('--bert_embedding_dimension', type=int, default=768, help='Embedding dimension for the Bert Model')
    parser.add_argument('--eval_batch_size', type=int, default=200, help='Eval Batch Size for the Bert Model')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning Rate for fine-tuning the Bert Model')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of Epochs')
    parser.add_argument('--patience', type=int, default=2, help='Early Stopping Patience Value')
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":

    start_time_total = time.time()
    args = parse_args()
    args.cuda = True
    if args.train:
        if not args.training_file:
            print("--training_file argument should be provided if you need to train the classifier")
            sys.exit(0)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: you have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed_all(42)

    assert os.path.exists(curr_dir + "/tmp/token_dict.pkl"), 'ERROR: Please run frequent_ngram_gen.py before running this file! '
    assert os.path.exists(curr_dir + "/tmp/phrase_dict.pkl"), 'ERROR: Please run frequent_ngram_gen.py before running this file! '

    with open(curr_dir + "/tmp/token_dict.pkl", 'rb') as f:
        token_obj = pickle.load(f)

    with open(curr_dir + "/tmp/phrase_dict.pkl",'rb') as f:
        phrase_obj = pickle.load(f)

    train_phrases = list()
    train_phrases_string = list()
    train_phrases_labels = list()
    test_phrases = list()
    test_phrases_string = list()
    eval_batch_size = args.eval_batch_size
    if args.train:

        pretrained_weights = args.pretrained_weights
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
        embedding_dim = args.bert_embedding_dimension
        print("Loading labeled data")
        assert os.path.exists(args.training_file), 'ERROR: Labeled file not found! Please place the labeled file in the directory mentioned and try again.'
        train_phrases_raw = OrderedDict()
        with open(args.training_file) as f:
            content = f.readlines()
            for line in content:
                line = line.strip()
                phrase, label = line.split("\t")
                phrase = phrase.lower()
                label = label.lower()
                if phrase in phrase_obj.phrase_stoi:
                    train_phrases_raw[phrase_obj.phrase_stoi[phrase]] = label
                    train_phrases.append(phrase_obj.phrase_stoi[phrase])
                    train_phrases_string.append(phrase)
                    train_phrases_labels.append(label)

        for phrase, val in phrase_obj.phrase_stoi.items():
            if val not in train_phrases_raw:
                test_phrases.append(val)
                test_phrases_string.append(phrase)

        label_raw = train_phrases_labels
        label_encoded = label_encoder.fit_transform(label_raw)

        label_one_hot = one_hot_encoder.fit_transform(label_encoded.reshape(-1, 1)).toarray()
        label = torch.from_numpy(label_one_hot)
        train_sentences = ["[CLS]" + sentence + " [SEP]" for sentence in train_phrases_string]
        train_tokenized_texts = [tokenizer.tokenize(sent) for sent in train_sentences]

        MAX_LEN = 10
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in train_tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        attention_masks = []

        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)

        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, label_encoded, random_state=42, test_size=0.2)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.2)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        batch_size = 32

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_samples = RandomSampler(train_data)
        train_data_loader = DataLoader(train_data, sampler=train_samples, batch_size=batch_size, drop_last=False)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, drop_last=False)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.train:

        model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=2, output_hidden_states=True)
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'decay_rate': 0.01
             },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'decay_rate': 0.0
             }
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=.1)

        epochs = args.num_epochs
        is_dnn_best = False
        dnn_min_loss = float("inf")
        max_patience = args.patience
        for _ in trange(epochs, desc="Epoch"):

            model.train()
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(train_data_loader):
                batch = tuple(t.to(device) for t in batch)
                b_inputs_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()

                outputs = model(b_inputs_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]
                loss.backward()

                optimizer.step()

                tr_loss += loss.item()
                nb_tr_steps += 1

            print("Train Loss: ", tr_loss/nb_tr_steps)

            model.eval()

            val_loss = 0
            nb_val_steps = 0
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_inputs_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = model(b_inputs_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss, logits = outputs[:2]

                val_loss += loss.item()
                nb_val_steps += 1

            print("Validation Loss: ", val_loss/nb_val_steps)
            is_dnn_best = (val_loss/nb_val_steps) < dnn_min_loss
            dnn_min_loss = min(val_loss, dnn_min_loss)

            if is_dnn_best:
                patience = 1
                model_to_save = model.module if hasattr(model, 'module') else model

                torch.save(model_to_save.state_dict(), args.output_model)
                model_to_save.config.to_json_file(args.output_config)
                tokenizer.save_vocabulary(args.output_vocab)
            else:
                patience += 1

            if patience >= max_patience:
                print("Patience limit reached. Stopping training.")
                break

    else:
        for phrase, val in phrase_obj.phrase_stoi.items():
            test_phrases.append(val)
            test_phrases_string.append(phrase)

    config = BertConfig.from_json_file(args.output_config)
    model = BertForSequenceClassification(config)
    state_dict = torch.load(args.output_model)
    model.load_state_dict(state_dict)
    model.to(device)
    tokenizer = BertTokenizer(args.output_vocab, do_lower_case=True)
    model.eval()
    preds = []

    test_sentences = ["[CLS]" + sentence + " [SEP]" for sentence in test_phrases_string]
    test_tokenized_texts = [tokenizer.tokenize(sent) for sent in test_sentences]

    MAX_LEN = 10
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in test_tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    test_inputs = torch.tensor(input_ids)
    test_labels = torch.zeros((len(test_inputs), 1))
    test_masks = torch.tensor(attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size, drop_last=False)

    if not os.path.exists(curr_dir + '/output/'):
        os.makedirs(curr_dir + '/output/')
    output_phrases_raw = []

    embedding_dict = dict()
    j = 0
    for batch in test_dataloader:
        j += 1
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=None)
            logits = outputs[0]

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    for j in range(len(test_phrases)):
        if preds[j] == 1:
            output_phrases_raw.append(test_phrases_string[j])

    # output_phrases = OrderedDict(sorted(output_phrases_raw.items(), key= lambda x: x[1], reverse=True))
    with open(args.output_file, "w") as f:
        for key in output_phrases_raw:
            f.write(key + '\n')

    print("Extraction completed!")
    end_time_total = time.time()
    print("Total time for complete process:", end_time_total - start_time_total, " seconds")
