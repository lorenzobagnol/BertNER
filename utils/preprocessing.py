import torch
from transformers import BertTokenizerFast

def align_sentence_label(tokenized_sentence, sentence_labels, labels_to_ids, label_all_tokens):

    word_ids = tokenized_sentence.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[sentence_labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[sentence_labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, X, Y, labels_to_ids):

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        lb = Y
        txt = X
        self.texts = [tokenizer(i, padding='max_length', max_length = 64, truncation=True, return_tensors="pt", is_split_into_words=True) for i in txt]
        self.labels = [align_sentence_label(i, j, labels_to_ids, True) for i,j in zip(self.texts, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels