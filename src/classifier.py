import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, input_dim, label_dim):
        super(Classifier, self).__init__()
        self.finalClassifier = nn.Linear(input_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.BCELoss = nn.BCELoss()

    def forward(self, feature):

        output = self.sigmoid(self.finalClassifier(feature))
        return output

    def loss(self, outputs, labels):
        return self.BCELoss(outputs, labels)
