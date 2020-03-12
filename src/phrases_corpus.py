import os
from collections import OrderedDict
from ordered_default_dict import DefaultListOrderedDict
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
curr_dir = os.getcwd()


class Phrase:

    def __init__(self, token_obj, len_threshold, freq_threshold):

        if not os.path.exists(curr_dir + '/tmp/'):
            os.makedirs(curr_dir + '/tmp/')

        tmp_all_phrases_filename = 'frequent_all_phrases.txt'
        all_phrases = open(curr_dir + '/tmp/' + tmp_all_phrases_filename, 'w')
        tmp_multi_word_phrases_filename = 'frequent_multi_word_phrases.txt'
        multi_word_phrases = open(curr_dir + '/tmp/' + tmp_multi_word_phrases_filename, 'w')
        self.phrase_stoi = OrderedDict()
        self.phrase_itos = list()
        self.phrase_positions = DefaultListOrderedDict()
        self.phrase_tokens = DefaultListOrderedDict()
        self.phrase_stats = DefaultListOrderedDict()
        num_tokens = len(token_obj.all_doc_tokens)
        for i in range(num_tokens):
            token = token_obj.all_doc_tokens[i]
            if token == token_obj.stoi['$']:
                continue
            self.phrase_positions[token_obj.itos[int(token)]].append(i)

        phrase_length = 1
        while len(self.phrase_positions) > 0:
            if phrase_length > len_threshold:
                break
            temp_dict = DefaultListOrderedDict()
            phrase_length += 1
            for token, positions in self.phrase_positions.items():
                freq = len(positions)
                if freq >= freq_threshold:
                    all_phrases.write(token + "," + str(len(positions)) + "\n")
                    for i in positions:
                        if i+1 < num_tokens:
                            if token_obj.all_doc_tokens[i+1] == token_obj.stoi['$']:
                                continue
                            new_phrase = token + " " + token_obj.itos[int(token_obj.all_doc_tokens[i+1])]
                            temp_dict[new_phrase].append(i+1)
            self.phrase_positions.clear()
            self.phrase_positions = temp_dict

        all_phrases.close()
        self.phrase_positions = DefaultListOrderedDict()
        with open(curr_dir + '/tmp/' + tmp_all_phrases_filename) as f:
            content = f.readlines()
            for line in content:
                line = line.strip()
                if len(line) > 0:
                    phrase, freq = line.split(",")
                    self.phrase_positions[phrase] = freq

        self.phrase_num = 0
        for phrase, positions in self.phrase_positions.items():
            feature_bracket = 1 if phrase[0] == '(' and phrase[len(phrase) - 1] == ')' else 0
            phrase = re.sub(r'\)\s{0,1}', '', phrase)
            phrase = re.sub(r'\(\s{0,1}', '', phrase)
            phrase = phrase.strip()
            tokens = phrase.split()
            if len(tokens) > 1 and tokens[len(tokens) - 1] not in stop_words:
                for i in range(len(tokens)):
                    self.phrase_tokens[self.phrase_num].append(token_obj.stoi[tokens[i]])
                self.phrase_stats[self.phrase_num].append(feature_bracket)
                self.phrase_stoi[phrase] = self.phrase_num
                self.phrase_itos.append(phrase)
                self.phrase_stats[self.phrase_num].append(positions)
                self.phrase_num += 1

                multi_word_phrases.write(phrase + "," + str(positions) + "\n")

        multi_word_phrases.close()
        print("# of multi-word frequent phrases:", self.phrase_num)

        self.phrase_positions.clear()

    def add_phrase(self, phrase_val, token_obj):

        tokens = phrase_val.split()
        pos = len(self.phrase_stoi) + 1
        if len(tokens) > 1 and tokens[len(tokens) - 1] not in stop_words:
            for t in tokens:
                if t not in token_obj.stoi:
                    token_obj.add_token(t)
                self.phrase_tokens[pos].append(token_obj.stoi[t])
            if '(' == phrase_val[0] and ')' == phrase_val[len(tokens) - 1]:
                self.phrase_stats[pos].append(1)
            else:
                self.phrase_stats[pos].append(0)
            phrase_val = re.sub(r'\)\s{0,1}', '', phrase_val)
            phrase_val = re.sub(r'\(\s{0,1}', '', phrase_val)
            self.phrase_stoi[phrase_val] = pos
            self.phrase_itos.append(phrase_val)
            self.phrase_stats[pos].append(1)

            self.phrase_num += 1
            return True
        else:
            return False
