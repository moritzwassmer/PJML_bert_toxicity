# Project Name: PJML_bert_toxicity

## Description:
This project has been constructed for the course `Project Machine Learning` in Winter 2023/2024. We implemented BERT (Bidirectional Encoder Representations from Transformers) from scratch to detect toxic content in text. 
Below we provide an overview over our approach and results. For more details, read the [report 1](reports/Project_Machine_Learning_MS1.pdf), [report 2](reports/Project_Machine_Learning_MS2.pdf), and [report 3](reports/Project_Machine_Learning_MS3.pdf).

### BERT


### Data
We used the data from the jigsaw [kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
The task at hand is a multi-label binary classification task, i.e. a comment can have multiple toxicity labels.


### Hyperparameteroptimization
We perform Hyperparameteroptimization to find an 'optimal' model 

### Explainable AI (XAI) 
We use Integrated Gradients to 

## Installation:
1. Clone the repository: `git clone https://github.com/your-username/PJML_bert_toxicity.git`
2. Navigate to the project directory: `cd PJML_bert_toxicity`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage:
1. Download the toxic comment dataset and place it in folder `in/toxic_comment`
[Dataset download](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
2. Configure `params.py` according to your needs
3. Train and evaluate BERT by using `main.py`
4. Explore predictions by running the `explain_str.ipynb` notebook
