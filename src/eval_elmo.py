from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
import pickle
import os
from collections import OrderedDict
import torch
import classifier
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import recall_score, f1_score, precision_score
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
label_encoder = LabelEncoder()
min_max_scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder()

curr_dir = os.getcwd()


def parse_args():

    parser = argparse.ArgumentParser(description='Clinical Phrase Extraction')
    parser.add_argument('--testing_file', type=str, default='../data/testing_set.txt', help='Name and Location of the testing file with phrases')
    parser.add_argument('--training_file', type=str, default='../data/training_set.txt', help='Name and Location of the training file with phrases')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--classifier_model', type=str, default='../model/classifier_test.pth', help='Name and Location of classifier model')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch Size for Classifier')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate for the Classifier')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of Epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early Stopping Patience Value')
    parser.add_argument('--elmo_dimensions', type=float, default=1024, help='Embedding dimensions for ELMo')
    parser.add_argument('--options_file', type=str, default='../model/biomed_elmo_options.json', help='Options file for ELMo')
    parser.add_argument('--weights_file', type=str, default='../model/biomed_elmo_weights.hdf5', help='Weights file for ELMo')
    parsed_args = parser.parse_args()
    return parsed_args


def calculate_metrics(predicted_list, actual_list):

    acc = ((predicted_list == actual_list).sum())/len(actual_list)
    f1 = f1_score(y_true=actual_list, y_pred=predicted_list, average='macro')
    recall = recall_score(y_true=actual_list, y_pred=predicted_list, average='macro')
    precision = precision_score(y_true=actual_list, y_pred=predicted_list, average='macro')
    return acc, f1, recall, precision


if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: you have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed_all(42)

    assert os.path.exists("../tmp/token_dict.pkl"), 'ERROR: Please run frequent_ngram_gen.py before running this file! '
    assert os.path.exists("../tmp/phrase_dict.pkl"), 'ERROR: Please run frequent_ngram_gen.py before running this file! '

    with open('../tmp/token_dict.pkl', 'rb') as f:
        token_obj = pickle.load(f)

    with open('../tmp/phrase_dict.pkl', 'rb') as f:
        phrase_obj = pickle.load(f)

    options_file = args.options_file
    weight_file = args.weights_file

    # use batch_to_ids to convert sentences to character ids
    train_phrases_raw = OrderedDict()
    train_phrases = list()
    train_phrases_string = list()
    train_phrases_labels = list()

    assert os.path.exists(args.training_file), 'ERROR: Training file not found! Please place the testing file in the directory mentioned and try again.'
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

    sentences = [phrase.split() for phrase in train_phrases_string]
    character_ids = batch_to_ids(sentences)

    label_raw = train_phrases_labels
    label_encoded = label_encoder.fit_transform(label_raw)
    label_one_hot = one_hot_encoder.fit_transform(label_encoded.reshape(-1, 1)).toarray()
    label = torch.from_numpy(label_one_hot)

    length = len(train_phrases)
    train_length = int(0.8 * length)
    val_length = length - train_length

    train_features, val_features = torch.split(character_ids, [train_length, val_length], dim=0)
    train_label, val_label = torch.split(label, [train_length, val_length], dim=0)

    input_dim = args.elmo_dimensions
    label_dim = 2

    model = classifier.Classifier(input_dim, label_dim, options_file, weight_file)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=False, patience=10, factor=0.5)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    is_dnn_best = False
    dnn_min_loss = float("inf")
    max_patience = args.patience
    batch_size = args.batch_size
    patience = 0
    train_dataset = TensorDataset(train_features, train_label)
    val_dataset = TensorDataset(val_features, val_label)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for epoch in range(args.num_epochs):
        for i, (feature, label) in enumerate(train_data_loader):
            model.train()
            features = Variable(feature).to(device)
            label = Variable(label).to(device)
            outputs = model(features)
            loss = model.loss(outputs, label.float())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        val_loss = 0.0
        val_steps = len(val_data_loader)
        for i, (feature, label) in enumerate(val_data_loader):
            with torch.no_grad():
                val_features = feature.to(device)
                label = label.to(device)
                val_output = model(val_features)
                val_loss += model.loss(val_output, label.float())

        if epoch % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Val_Loss:{:.4f}' .format(epoch+1, args.num_epochs, loss.item(), val_loss.item()/val_steps))
        lr_scheduler.step(val_loss)
        is_dnn_best = val_loss < dnn_min_loss
        dnn_min_loss = min(val_loss, dnn_min_loss)

        if is_dnn_best:
            patience = 1
            checkpoint = {'input_dim': input_dim,
                          'label_dim': label_dim,
                          'device': device,
                          'state_dict': model.state_dict()}
            torch.save(checkpoint, "../model/classifier_test.pth")
        else:
            patience += 1

        if patience >= max_patience:
            print("Patience limit reached. Stopping training.")
            break

    phrases = list()
    phrases_string = list()
    phrases_label = list()
    phrase_excluded = list()
    phrase_excluded_label = list()
    eval_batch_size = args.batch_size
    print("Loading data")
    assert os.path.exists(args.testing_file), 'ERROR: Testing file not found! Please place the testing file in the directory mentioned and try again.'
    print("Phrase Obj Length:", len(phrase_obj.phrase_itos))
    phrases_raw = OrderedDict()
    with open(args.testing_file) as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            phrase, label = line.split("\t")
            if phrase in phrase_obj.phrase_stoi:
                phrases_raw[phrase_obj.phrase_stoi[phrase]] = label
                phrases_label.append(label)
                phrases.append(phrase_obj.phrase_stoi[phrase])
                phrases_string.append(phrase)
            else:
                phrase = phrase.strip()
                tokens = phrase.split()
                if len(tokens) > 1:
                    if phrase_obj.add_phrase(phrase, token_obj):
                        phrases_raw[phrase_obj.phrase_stoi[phrase]] = label
                        phrases.append(phrase_obj.phrase_stoi[phrase])
                        phrases_string.append(phrase)
                        phrases_label.append(label)
                    else:
                        phrase_excluded.append(phrase)
                        phrase_excluded_label.append(phrase)

    label_raw = phrases_label
    label_encoded = label_encoder.fit_transform(label_raw)

    sentences = [phrase.split() for phrase in phrases_string]
    character_ids = batch_to_ids(sentences)

    test_labels = torch.zeros((len(features), 1))
    model = classifier.Classifier(args.elmo_dimensions, label_dim, options_file, weight_file)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        checkpoint = torch.load(args.classifier_model)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        device = torch.device("cpu")
        checkpoint = torch.load(args.classifier_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    test_dataset = TensorDataset(character_ids, test_labels)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    predicted_scores = torch.zeros(len(features))
    with torch.no_grad():
        loss = 0
        for i, (features, label) in enumerate(test_data_loader):
            feature = features.to(device)
            label = label
            output = model(feature)
            _, predicted = torch.max(output, 1)
            for j in range(output.size(0)):
                predicted_scores[i*batch_size + j] = predicted[j].item()

    predicted_scores = predicted_scores.reshape(-1).long()
    predicted = predicted_scores.cpu().numpy()

    acc, f1, recall, precision = calculate_metrics(predicted, label_encoded)

    print("Accuracy: ", acc)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
