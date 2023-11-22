import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from params import *


def generate_wordcloud(text, Title, max_words=20):
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, stopwords=set(STOPWORDS), background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(Title)

def show(output_folder=None, graph_name=None):
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Graph 1
    if graph_name == 'dstr_toxic':
        label_counts_total = np.zeros(len(ORDER_LABELS))

        for batch in train_loader:
            labels = batch["labels"].cpu().numpy()
            label_counts_total += labels.sum(axis=0)

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=label_counts_total, y=ORDER_LABELS, palette='viridis')

        for i, v in enumerate(label_counts_total):
            ax.text(v + 0.5, i, str(v), color='black', ha='left', va='center')

        plt.xlabel('Number of Occurrences')
        plt.ylabel('Labels')
        plt.title('Distribution of Toxic Label Occurrences')

    # Graph 2
    elif graph_name == 'wrdcloud_clean':
        non_toxic_text = ""
        for batch in train_loader:
            labels = batch["labels"]
            non_toxic_indices = (labels.sum(dim=1) == 0).nonzero().squeeze()
            non_toxic_texts = [tokenizer.decode(batch["input"][i], skip_special_tokens=True) for i in non_toxic_indices]
            non_toxic_text += " ".join(non_toxic_texts)

        generate_wordcloud(non_toxic_text, "Word Cloud for Non-Toxic Comments")

    # Graph 3
    elif graph_name == 'wrdcloud_toxic':
        toxic_text = ""
        for batch in train_loader:
            labels = batch["labels"]
            toxic_indices = (labels.sum(dim=1) > 0).nonzero().squeeze()
            toxic_texts = [tokenizer.decode(batch["input"][i], skip_special_tokens=True) for i in toxic_indices]
            toxic_text += " ".join(toxic_texts)

        generate_wordcloud(toxic_text, "Word Cloud for Toxic Comments")

    # Graph 4
    elif graph_name == 'length_per_label':
        label_lengths = {label: [] for label in ORDER_LABELS}
        clean_lengths = []

        for batch in train_loader:
            labels = batch["labels"]
            input_texts = batch["input"]
            comment_lengths = [len(tokenizer.decode(text, skip_special_tokens=True)) for text in input_texts]

            for i, label in enumerate(ORDER_LABELS):
                label_indices = torch.where(labels[:, i] == 1)[0]
                label_lengths[label].extend([comment_lengths[i] for i in label_indices])

            clean_indices = torch.where(labels.sum(dim=1) == 0)[0]
            clean_lengths.extend([comment_lengths[i] for i in clean_indices])

        average_label_lengths = {label: sum(lengths) / len(lengths) if lengths else 0 for label, lengths in label_lengths.items()}
        average_clean_length = sum(clean_lengths) / len(clean_lengths) if clean_lengths else 0

        data = {
            "Label": list(average_label_lengths.keys()) + ["Clean"],
            "Average Length (Characters)": list(average_label_lengths.values()) + [average_clean_length]
        }

        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x="Average Length (Characters)", y="Label", data=df, palette='viridis')

        for i, v in enumerate(df["Average Length (Characters)"]):
            ax.text(v + 0.5, i, str(round(v, 2)), color='black', ha='left', va='center')

        plt.xlabel('Average Comment Length (Characters)')
        plt.ylabel('Label')
        plt.title('Average Comment Length for Each Label')

    # Graph 5
    elif graph_name == 'dstr_length':
        all_comment_lengths = []

        for batch in train_loader:
            input_texts = batch["input"]
            comment_lengths = [len(tokenizer.decode(text, skip_special_tokens=True)) for text in input_texts]
            all_comment_lengths.extend(comment_lengths)

        all_comment_lengths = torch.tensor(all_comment_lengths).numpy()
        plt.figure(figsize=(10, 5))
        ax = sns.boxplot(x=all_comment_lengths, palette='viridis')

        plt.xlabel('Number of Tokens')
        plt.title('Distribution of Comment Lengths')

    else:
        print("Invalid graph name")

    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f'graph_{graph_name}.png'))
    else:
        plt.show()

# Example usage:
# show(graph_name= "dstr_toxic")  # Plots without saving
# show(output_folder='output_folder', graph_name="dstr_toxic")  # Plots and saves in 'output_folder'
