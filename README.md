# BERT toxic comment detection

## Description:
This project has been constructed for the course `Project Machine Learning` in Winter 2023/2024. We implemented BERT (Bidirectional Encoder Representations from Transformers) from scratch to detect toxic content in text. 
Below we provide an overview over our approach and results. For more details, read the [report 1](reports/Project_Machine_Learning_MS1.pdf), [report 2](reports/Project_Machine_Learning_MS2.pdf), and [report 3](reports/Project_Machine_Learning_MS3.pdf).

### BERT
We use [BERT](https://aclanthology.org/N19-1423/) as a neural network architecture. The Architecture was implemented from scratch but we loaded the pretrained weights to save resources. We only fine-tune on the toxicity task.

![image](https://github.com/user-attachments/assets/3849d9d3-65fa-4196-8969-2e53f5e5fafe)

### Data
We used the data from the jigsaw [kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
The task at hand is a multi-label binary classification task, i.e. a comment can have multiple toxicity labels (toxic, severe toxic, obscene, threat, insult, identity hate)

![image](https://github.com/user-attachments/assets/6ab77ba6-cedb-4c2c-a9ee-00d60482c46a)

### Hyperparameters

For details on the hyperparameters, view the code or review the reports. The most important hyperparameters are:

#### BERT Configuration
(H = 768, L = 12, A = 12, S = 512)

#### loss function
To tackle Class imbalance and to handle the multi label task, we adapt the standard binary classification loss the following way:

$\ell_{n, c} = -\left[ p_c y_{n, c} \cdot \log\sigma(x_{n, c}) + (1 - y_{n, c}) \cdot \log(1 - \sigma(x_{n, c})) \right]$

### XAI and Fairness Evaluation
We use Integrated Gradients to interpret the results using [Captum](https://captum.ai/)

![image](https://github.com/user-attachments/assets/df5151f7-b748-4e47-a6a0-b05a9639ff49)

We analyze wether the decisions are fair and biases. In the image below, a comment is classified as `identity hate` even though the comment is not toxic in any way. There is a bias for the use of certain words such as `jew` towards usage in toxic contexts.

![image](https://github.com/user-attachments/assets/09179408-ffbe-4402-9c03-f724051a8af7)

We analyze performance scores across different identities:
![image](https://github.com/user-attachments/assets/d9ffcfb2-9fe2-45e4-b00d-fe96e746e757)

### Performance Evaluation

![image](https://github.com/user-attachments/assets/0b3e199d-368c-46ba-9594-22bb0fdde294)


![image](https://github.com/user-attachments/assets/8e95c257-c624-457b-91d4-d2f353bf377a)


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
