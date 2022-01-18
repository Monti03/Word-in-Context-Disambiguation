# Word-in-Context Disambiguation
Word-in-Context Disambiguation is the task of addressing the disambiguation of polysemous words, without relying on a fixed inventory of word senses. In this report I am going to propose a model based on a LSTM to solve this problem.

## Approach
The model that I am proposing takes as input the two sentences (stored as tensors of indexes) that are associated to the two sequences of embeddings relative to the terms inside the phrases through an Embedding layer. The two sequences of embeddings pass through the same BiLSTM (one sentence per time). From the output of the BiLSTM we take the tensor relative to the target term `term_i` (where `i` is `1` or `2` representing the sentence), the forward output of the last term `last_i` and the backward output of the first term `first_i`. After doing this for both the sentences we have to aggregate the results: since I wanted a network whose result is independent from the order of the sentences to compare I decided to do the following: I concatenate `abs(term_1 − term_2)`, `abs(first_1−first_2)`, `abs(last_1−last_2)`, `cossim(term_1, term_2)`, `cossim(first_1, first_2)`, `cossim(last_1,last_2)`, where `abs(x − y)` is the tensor equal to the difference of x and y where each element is chosen in its absolute value and `cossim(x,y)` is the cosine similarity of the two tensors. This concatenation than is passed through two fully connected layers. The model can be seen in the image below.

![image](https://user-images.githubusercontent.com/38753416/149948762-4815dc06-5b4d-4682-b317-d838dc9ceead.png)


#### Instructor
* **Roberto Navigli**
	* Webpage: http://wwwusers.di.uniroma1.it/~navigli/

#### Teaching Assistants
* **Cesare Campagnano**
* **Pere-Lluís Huguet Cabot**

#### Course Info
* http://naviglinlp.blogspot.com/

## Requirements

* Ubuntu distribution
	* Either 19.10 or the current LTS are perfectly fine
	* If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Notes
Unless otherwise stated, all commands here are expected to be run from the root directory of this project

## Setup Environment

As mentioned in the slides, differently from previous years, this year we will be using Docker to remove any issue pertaining your code runnability. If test.sh runs
on your machine (and you do not edit any uneditable file), it will run on ours as well; we cannot stress enough this point.

Please note that, if it turns out it does not run on our side, and yet you claim it run on yours, the **only explanation** would be that you edited restricted files, 
messing up with the environment reproducibility: regardless of whether or not your code actually runs on your machine, if it does not run on ours, 
you will be failed automatically. **Only edit the allowed files**.

To run *test.sh*, we need to perform two additional steps:
* Install Docker
* Setup a client

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding. For those who might be
unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw1 python=3.7
conda activate nlp2021-hw1
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw1
bash test.sh data/dev.jsonl
```

Actually, you can replace *data/dev.jsonl* to point to a different file, as far as the target file has the same format.

If you hadn't changed *hw1/stud/model.py* yet when you run test.sh, the scores you just saw describe how a random baseline
behaves. To have *test.sh* evaluate your model, follow the instructions in the slide.
