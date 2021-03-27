from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup
import torch
from dataset.encode import encode_paragraph
from dataset.loader import SciFactJointDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_rationale_label, flatten
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os
from pathlib import Path


def evaluation_joint(model, dataset, tokenizer):
    model.eval()
    label = []
    output = []
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            logits = model(**encoded_dict)[0]
            out = torch.argmax(logits.cpu(), dim=-1).detach().numpy().tolist()
            # rationale_out = torch.argmax(rationale_score.cpu(), dim=-1).detach().numpy().tolist()

            label.extend(batch['abstract_label'])
            output.extend(out)

    return {
        'abstract_macro_f1': f1_score(label, output, zero_division=0, average='micro'),
        'abstract_f1': tuple(f1_score(label, output, zero_division=0, average=None)),
        'abstract_precision': precision_score(label, output, zero_division=0, average='micro'),
        'abstract_recall': recall_score(label, output, zero_division=0, average='micro'),
    }

def main():
    claim_train_path = '../data/train_data.jsonl'
    claim_dev_path = '../data/dev_data.jsonl'
    claim_test_path = '../data/claims_dev_retrieved_tfidf.jsonl'
    corpus_path = '../data/corpus.jsonl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert = 'allenai/scibert_scivocab_cased'
    tokenizer = BertTokenizer.from_pretrained(bert)
    train_set = SciFactJointDataset(corpus_path, claim_train_path, sep_token=tokenizer.sep_token, k=12)
    dev_set = SciFactJointDataset(corpus_path, claim_dev_path, sep_token=tokenizer.sep_token, k=12)
    test_set = SciFactJointDataset(corpus_path, claim_test_path,
                                   sep_token=tokenizer.sep_token, k=30, train=False)
    batch_size = 1
    epochs = 4
    model = BertForSequenceClassification.from_pretrained(bert, num_labels=3).to(device)

    optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': 1e-5},
        ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs)
    model.train()
    best_f1 = 0
    checkpoint = os.path.join('model/', f'SciBert_checkpoint')
    for epoch in range(epochs):
        t = tqdm(DataLoader(train_set, batch_size=batch_size, shuffle=False))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            output = model(**encoded_dict, labels=batch['abstract_label'].to(device))
            loss = output[0]
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')

        scheduler.step()
        train_score = evaluation_joint(model, train_set, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score)
        dev_score = evaluation_joint(model, dev_set, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score)
        if dev_score['abstract_macro_f1'] > best_f1:
            best_f1 = dev_score['abstract_macro_f1']
            best_model = model
    if not Path(checkpoint).exists():
        os.makedirs(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    best_model.save_pretrained(checkpoint)


if __name__ == "__main__":
    main()






