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

    def __init__(self, X, Y, dict, label_all_tokens=False):

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        lb = Y
        txt = X
        self.texts = [tokenizer(i, padding='max_length', max_length = 64, truncation=True, return_tensors="pt", is_split_into_words=True) for i in txt]
        self.labels = [align_sentence_label(i, j, dict.labels_to_ids, label_all_tokens) for i,j in zip(self.texts, lb)]

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
    
class Dictionary:
    
    def __init__(self, list_labels):
        
        unique_labels = set()
        for label_sent in list_labels:
            for label in label_sent:
                unique_labels.add(label)
        self.unique_labels=unique_labels
        self.labels_to_ids = {k: v for v, k in enumerate(sorted(self.unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(self.unique_labels))}   
    
    def transform_labels_to_ids(self, label_list):        
        return [self.labels_to_ids[a] for a in label_list]
    
    def transform_ids_to_labels(self, ids_list):        
        return [self.ids_to_labels[a] for a in ids_list]
    
    
    
def remove_BIOES_tags(label_list):
    cleaned_sentence=list()
    for tag in label_list:
        if tag.startswith('O'):
            cleaned_sentence.append(tag)
        else: 
            cleaned_sentence.append(tag[2:])
    return cleaned_sentence