import numpy as np
from typing import List, Tuple, Dict, Optional
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter, defaultdict

from model import Model
import torch
from torch import nn

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    
    def __init__(self, device):
        # download nltk data
        nltk.download('stopwords')
        nltk.download('punkt')
        # some constants
        self.device = device
        self.WEIGHT = 0
        self.EMBEDDING_DIMENSION = 300
        self.STOPWORDS = stopwords.words()
        self.STOPWORDS_WEIGHT = 0.1
        # define, load and move to device the model
        self.model = Classifier(self.EMBEDDING_DIMENSION, n_hidden1=512, n_hidden2=256, dropout=0.5)
        self.model.load_state_dict(torch.load("model/Download-data-734-HW1-word_level.pt"))
        self.model.to(self.device)
        self.model.eval() 
        # I will initialize the embeddings in the first predict
        self.embeddings = {}
    
        
    def preprocess_phrase(self, sentence:str, start:int, end:int):
        # get the original target term
        original_term = sentence[start:end].lower()
        
        # replace punctuation with space
        sentence = re.sub(r"[()\",\-—.;/:@<>#?!&$“”'’`–%\[\]−]+", " ", sentence) 
        # remove extra space
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = sentence.replace("'s", " 's")
        # strip and lowerize the obtained sentence
        sentence = sentence.strip().lower()
        
        # get the new position of the target term
        start_n = sentence.find(original_term)
        end_n = start_n + len(original_term)

        # check that the position is correct
        assert original_term == sentence[start_n:end_n] 

        return sentence, start_n, end_n

    # preprocess each phrase and return a list of dictionaries with the same structure
    # of the data passed to the predict method
    def preprocessing(self, entries: List[Dict])-> List[Dict]:
        preprocessed = []
        for entry in entries:
            
            # get preprocessed phrase and the position of the target term
            sentence1, start1, end1 = self.preprocess_phrase(entry["sentence1"], int(entry["start1"]), int(entry["end1"]))
            sentence2, start2, end2 = self.preprocess_phrase(entry["sentence2"], int(entry["start2"]), int(entry["end2"]))
            
            # copy the entry and replace senteces and positions of the target terms
            # in both the phrases
            preprocessed_entry = entry.copy()
            preprocessed_entry["sentence1"] = sentence1
            preprocessed_entry["sentence2"] = sentence2
            preprocessed_entry["start1"] = start1
            preprocessed_entry["end1"] = end1
            preprocessed_entry["start2"] = start2
            preprocessed_entry["end2"] = end2
            
            preprocessed.append(preprocessed_entry)

        return preprocessed

    # associate a phrase to a tensor by computing a weigted average of the embddins
    # of the terms in the phrase
    def phrase2vector(self, phrase: str, lemma:str):
        # split the phrase into tokens
        terms = phrase.split(" ")
        # start the phrase embedding as a zero tensor
        sum = torch.zeros(self.EMBEDDING_DIMENSION)

        # n -> num of non zero embeddings
        n = 0
        for term in terms:
            if (term in self.embeddings):
                # get the embedding of the term and turn it to tensor
                emb = self.embeddings[term]
                # update the counter of non zero embeddings
                n += 1
                # update the sum of embeddings of the phrase 
                # the target term is weighted with WEIGHT
                # the stopwords also are weighted with STOPWORDS_WEIGHT
                # the other term have weight = 1
                sum = sum + self.WEIGHT*emb if term == lemma else (sum + self.STOPWORDS_WEIGHT*emb if term in self.STOPWORDS else sum + emb)

            # in order to avoid possible errors in test phase
            # (for train and dev data is not necessary)
            n = 1 if n == 0 else n 
        return sum/n

    # initialize the data
    def _init_data(self, preprocessed):
        # iterate on the given file and build samples
        samples = []
        for entry in preprocessed:
            term_1 = entry["sentence1"][entry["start1"]: entry["end1"]]
            term_2 = entry["sentence2"][entry["start2"]: entry["end2"]]

            emb_1 = self.phrase2vector(entry["sentence1"], term_1)
            emb_2 = self.phrase2vector(entry["sentence2"], term_2)
            
            data = (emb_1, emb_2)

            samples.append(data)
        
        return samples

    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        
        if(self.embeddings == {}):
        
            # load the glove embeddings
            with open(f"model/glove.840B.300d.txt") as fin:
                for line in fin:
                    splitted_line = line.split(" ")
                    term = splitted_line[0]
                    embedding = splitted_line[1:]
                    embedding = torch.tensor(list(map(lambda x: float(x), embedding)))
                    self.embeddings[term] = embedding
        # preprocess data
        preprocessed = self.preprocessing(sentence_pairs)
        # get the phrase embeddings
        data = self._init_data(preprocessed)
        
        input_to_test_1 = data[0][0]
        input_to_test_2 = data[0][1]

        if(len(data) == 1):
            input_to_test_1 = input_to_test_1.unsqueeze(0)
            input_to_test_2 = input_to_test_2.unsqueeze(0)
        else:
            for x in data[1:]:
                input_to_test_1 = torch.vstack((input_to_test_1,x[0]))
                input_to_test_2 = torch.vstack((input_to_test_2,x[1]))
        
        # move data to device
        input_to_test_1.to(self.device)
        input_to_test_2.to(self.device)

        # predict 
        res = self.model(input_to_test_1, input_to_test_2)
        # round the predictions
        res = torch.round(res["pred"])
        
        res = res.tolist()
        if (len(data) == 1):
            res = [res]
        # return a list of True if 1, False if 0
        res = ["True" if x == 1 else "False" for x in res]

        return res

class Classifier(torch.nn.Module):

    def __init__(self, 
                 n_features: int, 
                 n_hidden1: int, 
                 n_hidden2: int, 
                 dropout:float):
        super().__init__()
        # the first linear layer is shared among the emebeddings of the two sentences
        self.lin1 = torch.nn.Linear(n_features, n_hidden1)
        # second lienar layer
        self.lin2 = torch.nn.Linear(n_hidden1, n_hidden2)
        # third one that creates the output
        self.lin3 = torch.nn.Linear(n_hidden2, 1)
        # dropout
        self.drop = torch.nn.Dropout(dropout)
        # batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden2)

        self.loss_fn = torch.nn.BCELoss()
        self.global_epoch = 0
    
    def forward(self, ph1: torch.Tensor, ph2: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict:
        
        # the embeddings of the two phrases pass through the same linear layer
        # so that changing the order does not change the result
        out1 = self.lin1(ph1)
        out1 = torch.relu(out1)
        # I use the batch normalization in order to speed up the learning
        out1 = self.bn1(out1)

        out2 = self.lin1(ph2)
        out2 = torch.relu(out2)
        out2 = self.bn1(out2)


        # element wise product of the two matrices
        # in order to merge the results of the first linear layer
        # of the two sentence embedding
        out = out1*out2    

        # fc layer with dropout
        res = self.lin2(out)
        res = torch.relu(res)
        res = self.drop(res)
        
        res = self.lin3(res)
        # pass results in 0-1 range
        res = torch.sigmoid(res)
        
        # remove a dimension
        res = res.squeeze()

        # save the predictions in order to return it
        result = {"pred": res}

        # compute loss and acc
        if y is not None:
            loss = self.loss(res, y)
            result["loss"] = loss
            result["acc"] = (torch.round(res) == y).float().sum()/len(y)
            result["correct"] = (torch.round(res) == y).float().sum()

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y) 