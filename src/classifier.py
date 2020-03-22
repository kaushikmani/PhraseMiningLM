import torch.nn as nn
from allennlp.modules.elmo import Elmo
import torch


class Classifier(nn.Module):

    def __init__(self, input_dim, label_dim, options_file, weight_file):
        super(Classifier, self).__init__()

        self.elmo = Elmo(options_file, weight_file, 3, dropout=0)
        self.finalClassifier = nn.Linear(input_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.BCELoss = nn.BCELoss()

    def forward(self, character_ids):

        elmo_out = self.elmo(character_ids)
        elmo_out = elmo_out['elmo_representations']
        elmo_stack = torch.stack(elmo_out)
        elmo_mean = torch.mean(elmo_stack, dim=0)
        elmo_mean = torch.mean(elmo_mean, dim=1)
        output = self.sigmoid(self.finalClassifier(elmo_mean))
        return output

    def loss(self, outputs, labels):
        return self.BCELoss(outputs, labels)
