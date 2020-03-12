import os
import math
from collections import OrderedDict
from ordered_default_dict import DefaultListOrderedDict


class Token:

    def __init__(self, file_name):

        assert os.path.exists(file_name), 'ERROR: Input file not found! Please place the input file in the directory mentioned and try again'
        file = open(file_name, 'r')
        self.all_doc_tokens = list()

        # 0 - freq , 1 - idf
        self.token_stats = DefaultListOrderedDict()
        self.stoi = OrderedDict()
        self.itos = list()

        self.token_num = 0
        self.stoi['$'] = self.token_num
        self.token_num += 1
        self.itos.append('$')

        self.num_docs = 0
        self.doc_token_count = OrderedDict()
        line_terminators = ".!?,;:\"[]"
        for line in file:
            self.num_docs += 1
            chars = []
            current_doc_tokens = {}
            for ch in line:
                pos = 0
                total = len(line)
                brackets = 0
                if ch == '(':
                    brackets += 1
                    chars.append(ch)
                elif ch == ')':
                    brackets -= 1
                    chars.append(ch)
                elif brackets == 0:
                    if ch.isalpha():
                        chars.append(ch.lower())
                    elif ch.isdigit():
                        chars.append(ch)
                    elif ch == '\\' or (ch == '.' and pos != total - 1 and line[pos+1].isdigit()):
                        chars.append(ch)
                    else:
                        if len(chars) > 0:
                            tok = ''.join(chars)
                            if tok not in self.stoi:
                                self.stoi[str(tok)] = self.token_num
                                self.itos.append(str(tok))
                                self.token_num += 1
                                self.token_stats[self.stoi[str(tok)]].append(1)
                            else:
                                self.token_stats[self.stoi[str(tok)]][0] += 1
                            self.all_doc_tokens.append(self.stoi[str(tok)])
                            current_doc_tokens[self.stoi[str(tok)]] = True
                            chars = []
                            pos += 1
                if ch in line_terminators:
                    self.all_doc_tokens.append(self.stoi['$'])
            if len(chars) > 0:
                tok = ''.join(chars)
                if tok not in self.stoi:
                    self.stoi[str(tok)] = self.token_num
                    self.itos.append(str(tok))
                    self.token_num += 1
                    self.token_stats[self.stoi[str(tok)]].append(1)
                else:
                    self.token_stats[self.stoi[str(tok)]][0] += 1
                self.all_doc_tokens.append(self.stoi[str(tok)])
                current_doc_tokens[self.stoi[str(tok)]] = True
                chars = []

            for t in current_doc_tokens:
                if t in self.doc_token_count:
                    self.doc_token_count[t] += 1
                else:
                    self.doc_token_count[t] = 1

        file.close()

        for tok, l in self.token_stats.items():
            self.token_stats[tok].append(max(math.log(self.num_docs/float(self.doc_token_count[tok])), 1e-10))

        print("Total # of tokens:", len(self.all_doc_tokens))
        print("# of unique tokens:", len(self.stoi))

    def add_token(self, token_val):

        self.stoi[token_val] = self.token_num
        self.itos.append(token_val)
        self.token_stats[self.token_num].append(1)
        self.token_stats[self.token_num].append(1e-10)
        self.token_num += 1
