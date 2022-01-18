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
        # I will initialize the embeddings at the first predict call
        self.embeddings = {}
    

    # preprocess a phrase and return the new preprocessed phrase
    # and the position of the target term   
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
        
        sentence = " ".join(word_tokenize(sentence))

        sentence = sentence.replace(original_term, "TARGET_TERM")

        # get the new position of the target term
        start_n = sentence.find("TARGET_TERM")
        end_n = start_n + len("TARGET_TERM")

        # check that the position is correct
        assert "TARGET_TERM" == sentence[start_n:end_n] 

        return sentence, start_n, end_n

    # preprocess each phrase in the dataset and return a list of dicts
    # like the one that are passed to the predict function
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

    # associate each phrase to the sequence of indeces of the terms inside the phrase
    def phrase2indices(self, review: str, word_index: Dict) -> torch.Tensor:
        return torch.tensor([word_index[word] for word in review.split(" ")], dtype=torch.long)

    # initialize the data so that can be passed to the collate function
    def _init_data(self, preprocessed):
        # iterate on the given file and build samples
        samples = []
        for entry in preprocessed:
            term_1 = entry["sentence1"][entry["start1"]: entry["end1"]]
            term_2 = entry["sentence2"][entry["start2"]: entry["end2"]]

            # get the indexes of the phrases
            indexes1 = self.phrase2indices(entry["sentence1"], self.word_index)
            indexes2 = self.phrase2indices(entry["sentence2"], self.word_index)
            
            # get the target term position
            term_1_index = entry["sentence1"].split(" ").index(term_1)
            term_2_index = entry["sentence2"].split(" ").index(term_2)

            # build the data sample as a tuple containing 
            # 1) a tuple with the list of indexes of the terms relative to the phrases (in order)
            # 2) the target term position of the two phrases
            # 3) the label
            data = ((indexes1, indexes2), (term_1_index, term_2_index))

            # append the sample
            samples.append(data)
        
        return samples

    # returns a list containing in position i the emebdding of the term 
    # relative to position i. in position 0 we have the pad token and in position 1
    # we have the unk token
    def get_vector_store(self):
        # word dict associates each term to a index inside the vectors_store
        self.word_index = dict()
        # vectors_store is a list containing in position i the emebdding of the term 
        # relative to position i. in position 0 we have the pad token and in position 1
        # we have the unk token
        vectors_store = []

        # pad token, index = 0
        vectors_store.append(torch.zeros(300))

        # unk token, index = 1
        vectors_store.append(torch.rand(300))

        for word, vector in self.embeddings.items():
            self.word_index[word] = len(vectors_store)
            vectors_store.append(vector)

        self.word_index = defaultdict(lambda: 1, self.word_index)  # default dict returns 1 (unk token) when unknown word
        return torch.stack(vectors_store)
    
    # prepare an input for the model
    def single_collate_fn(self, sentences: List[torch.Tensor])-> Tuple[torch.Tensor, torch.Tensor]:

        sentence_lengths = torch.tensor([sentence.size(0)-1 for sentence in sentences], dtype=torch.long)
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0) 

        return sentences, sentence_lengths
    
    # prepare all the inputs for the model
    def collate_fun(self, 
        entries: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        sentences1 = [entry[0][0] for entry in entries]
        sentences2 = [entry[0][1] for entry in entries]

        # get the padded sentences and the lenght of the sentences
        sentences1, sentence_lengths1 = self.single_collate_fn(sentences1)
        sentences2, sentence_lengths2 = self.single_collate_fn(sentences2)

        #get target term position in the sentences
        target_term_indexes1 = torch.tensor([entry[1][0] for entry in entries])
        target_term_indexes2 = torch.tensor([entry[1][1] for entry in entries])

        return (sentences1, sentences2), (sentence_lengths1, sentence_lengths2), (target_term_indexes1, target_term_indexes2)

    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        
        if(self.embeddings == {}):
            # load the glove embeddings
            with open(f"model/glove.840B.300d.txt") as fin:
                for line in fin:
                    term, *embedding = line.strip().split(" ")
                    if (len(embedding) == self.EMBEDDING_DIMENSION):
                        embedding = torch.tensor(list(map(lambda x: float(x), embedding)))  
                        self.embeddings[term] = embedding
            
            # define embedding for each target term
            self.embeddings["TARGET_TERM"] = torch.zeros(self.EMBEDDING_DIMENSION)
            
            self.vectors_store = self.get_vector_store() 
            # define, load and move to the device the model
            self.model = Classifier(n_hidden1=128, n_hidden2=32, dropout=0.5, vectors_store=self.vectors_store, device=self.device)
            self.model.load_state_dict(torch.load("model/HW1-RNN-70.pt"))
            self.model.to(self.device)
            # put the model in eval state
            self.model.eval()

        # preprocess the sentences
        preprocessed = self.preprocessing(sentence_pairs)
        # prepare the data for the collate fun
        data = self._init_data(preprocessed)
        # prepare the data for the model
        data = self.collate_fun(data)

        # unpack and move the data to the model
        sentences, sentence_lengths, target_term_indexes = data
        sentences1, sentences2 = [x.to(self.device) for x in sentences]
        sentence_lengths1, sentence_lengths2 = [x.to(self.device) for x in sentence_lengths]
        target_term_indexes1, target_term_indexes2 = [x.to(self.device) for x in target_term_indexes]

        # use the model to predict the results
        res = self.model(sentences1, sentences2, sentence_lengths1, sentence_lengths2, target_term_indexes1, target_term_indexes2)

        # round the output in order to obtain a tensor of zeros and ones
        res = torch.round(res["pred"])
        # get a list
        res = res.tolist()
        # convert 1 to True and 0 to False
        res = ["True" if x == 1 else "False" for x in res]

        return res

class Classifier(torch.nn.Module):

    def __init__(self,
                 n_hidden1: int,
                 n_hidden2: int,                 
                 dropout:float,
                 vectors_store: torch.Tensor,
                 device: str):
        super().__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(vectors_store)
        self.n_hidden1 = n_hidden1
        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=300, hidden_size=n_hidden1, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)

        self.bn1 = torch.nn.BatchNorm1d(num_features = n_hidden1*4+3)
        self.bn2 = torch.nn.BatchNorm1d(num_features = n_hidden2)

        # dropout
        self.drop = torch.nn.Dropout(dropout)

        # classification head
        # *4 since -> *2 cause of the concatenation 
        #          -> *2 cause of the bidirectional LSTM
        self.lin1 = torch.nn.Linear(n_hidden1*4+3, n_hidden2)
        self.lin2 = torch.nn.Linear(n_hidden2, 1)

        self.loss_fn = torch.nn.BCELoss()
        self.global_epoch = 0

        self.cos = cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.device = device
    
    def forward(
        self, 
        sentences1: torch.Tensor,  
        sentences2: torch.Tensor,  
        sentence_lengths1: torch.Tensor, 
        sentence_lengths2: torch.Tensor, 
        target_term_indexes1: torch.Tensor, 
        target_term_indexes2: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # embedding words from indices
        embeddings1 = self.embedding(sentences1)
        embeddings2 = self.embedding(sentences2)

        # recurrent encoding
        recurrent_out1 = self.rnn(embeddings1)
        
        # get the output per term in the paddes sentence        
        recurrent_out1 = recurrent_out1[0]
 
        # get the target term backward and forward output
        target_term_vectors1 = recurrent_out1[range(recurrent_out1.shape[0]), target_term_indexes1]
        # get the first term backward output
        start_sentence_vectors1 = recurrent_out1[range(recurrent_out1.shape[0]), [0]*len(sentence_lengths1), self.n_hidden1:]
        # get the last term forward output
        end_sentence_vectors1 = recurrent_out1[range(recurrent_out1.shape[0]), sentence_lengths1, :self.n_hidden1]

        # recurrent encoding for the second sentence
        recurrent_out2 = self.rnn(embeddings2)

        # get the output per term in the paddes sentence        
        recurrent_out2 = recurrent_out2[0]

        # get the target term backward and forward output
        target_term_vectors2 = recurrent_out2[range(recurrent_out2.shape[0]), target_term_indexes2]
        # get the first term backward output
        start_sentence_vectors2 = recurrent_out2[range(recurrent_out2.shape[0]), [0]*len(sentence_lengths2), self.n_hidden1:]
        # get the last term forward output
        end_sentence_vectors2 = recurrent_out2[range(recurrent_out2.shape[0]), sentence_lengths2, :self.n_hidden1]
        
        # save memory
        recurrent_out2 = None

        # compute cosine similarity
        cos_target = torch.unsqueeze(self.cos(target_term_vectors1, target_term_vectors2),1)
        cos_start_sentence = torch.unsqueeze(self.cos(start_sentence_vectors1, start_sentence_vectors2),1)
        cos_end_sentence = torch.unsqueeze(self.cos(end_sentence_vectors1, end_sentence_vectors2),1)
        
        # concatenate values
        summary_vectors = torch.hstack((
                                        torch.abs(target_term_vectors1-target_term_vectors2),
                                        torch.abs(start_sentence_vectors1-start_sentence_vectors2),
                                        torch.abs(end_sentence_vectors1-end_sentence_vectors2),
                                        cos_target, 
                                        cos_start_sentence, 
                                        cos_end_sentence))
        
        summary_vectors = self.drop(summary_vectors)
        
        # two fc layers
        out = self.lin1(summary_vectors)
        out = torch.relu(out)
        out = self.bn2(out)
        out = self.lin2(out).squeeze(1)

        logits = out
        pred = torch.sigmoid(logits)

        result = {'logits': logits, 'pred': pred}

        # compute loss
        if y is not None:
            loss = self.loss(pred, y)
            result["loss"] = loss
            result["acc"] = (torch.round(pred) == y).float().sum()/len(y)
            # number of correct predictions
            result["correct"] = (torch.round(pred) == y).float().sum()

        return result
    def loss(self, pred, y):
        return self.loss_fn(pred, y)