from typing import Callable
import torch

class netRatio:

    def __call__(
            self, classifier_nn) -> Callable:
        self.classifier_nn = classifier_nn

        return self.log_prob

    def log_prob(self, inputs='', context=''):
        dis =  torch.sigmoid(self.classifier_nn(torch.cat((context, inputs), dim=1)))
        return torch.log(dis / (1. - dis))