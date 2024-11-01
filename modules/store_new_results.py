from .train_apply import *
from .params import *
from .training import *
import pickle
import io
from .models import *
import pandas as pd
import csv

def generate_input_tensor(text):
    """
    Tokenizes the input text and generates a tensor suitable for model input.

    Args:
        text (str): Input text to be tokenized.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
        max_sequence_length (int): Maximum sequence length for padding/truncation.

    Returns:
        torch.Tensor: Input tensor suitable for model input.
    """
   
    tokens = TOKENIZER(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )


    input_tensor = tokens["input_ids"].to(DEVICE)

    return input_tensor

def predict_csv(input, model=None):
    
    if model is None:
        predictions = input
    else:
        predictions = torch.sigmoid(model.forward(input))
        predictions = predictions.cpu().detach().numpy()
        classes = torch.sigmoid(model.forward(input))
        classes = classes.round().cpu().detach().numpy()

    confidence_scores = []
    for i in range(predictions.shape[0]):
        entropy_sum = 0
        count = 0
        sample_confidence = []
        for j in range(predictions.shape[1]):
            confidence = 1.0 - shannon_entropy(predictions[i, j])
            #entropy_sum += shannon_entropy(predictions[i, j])
            #count += 1
            sample_confidence.append(confidence)
        #sample_confidence.append(1 - (entropy_sum / count))
        confidence_scores.append(sample_confidence)

    confidence_scores = np.array(confidence_scores,  dtype=predictions.dtype)
    return predictions, confidence_scores, classes

def write_to_csv_from_csv(filename, input, classes, predictions, confidence_scores):
    header = ['comment_text'] + [f'{label}_class' for label in ORDER_LABELS] + [f'{label}_prediction' for label in ORDER_LABELS] + [f'{label}_confidence' for label in ORDER_LABELS]
    
    with open(filename, 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if csvfile.tell() == 0:
            writer.writerow(header)
        
        for i in range(len(classes)):
            row = [input] + classes[i].tolist() + predictions[i].tolist() + confidence_scores[i].tolist()
            writer.writerow(row)


if __name__ == "__main__":
    model = Model()
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    with open(OUTPUT + '/model.pkl', 'rb') as file:
        model = CPU_Unpickler(file).load()
    model.eval()

    name = OUTPUT + '/output.csv'
    df = pd.read_csv(name)
    model.to(DEVICE)

    for index, row in df.iterrows():
        predictions, confidence_scores, classes = predict_csv(generate_input_tensor(row['comment_text']), model)
        write_to_csv_from_csv(OUTPUT + '/results_new_cluster.csv', row['comment_text'], classes, predictions, confidence_scores)


