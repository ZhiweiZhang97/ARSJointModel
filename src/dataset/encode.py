import torch
from typing import List
import numpy as np


def encode_sen_pair(tokenizer, claims: List[str], sentences: List[str]):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_dict = tokenizer.batch_encode_plus(
        list(zip(claims, sentences)),
        padding=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            list(zip(claims, sentences)),
            max_length=512,
            truncation_strategy='only_first',
            padding=True,
            return_tensors='pt')
    # encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def encode_sentence(tokenizer, sentences: List[str]):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_dict = tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            sentences,
            max_length=512,
            truncation_strategy='only_first',
            padding=True,
            return_tensors='pt')
    # encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def longest_first_truncation(sentences, truncate_length):
    '''
    :param sentences: sentence to be truncated
    :param truncate_length: truncate length
    :return: sentence after truncation
    '''
    sent_lens = [len(sent) for sent in sentences]
    while np.sum(sent_lens) > truncate_length:  # truncation the longest sentence
        max_position = np.argmax(sent_lens)
        sent_lens[max_position] -= 1
    return [sentence[:length] for sentence, length in zip(sentences, sent_lens)]


def truncate(input_ids, max_length, sep_token_id, pad_token_id):
    all_paragraphs = []
    for paragraph in input_ids:
        valid_paragraph = paragraph[paragraph != pad_token_id]
        if valid_paragraph.size(0) <= max_length:
            all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
        else:
            # position of sep_token
            sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
            idx_by_sentence = []
            prev_idx = 0
            for idx in sep_token_idx:  # delineate the sentence in the Claim&Abstract
                idx_by_sentence.append(paragraph[prev_idx:idx])
                prev_idx = idx
            # The last sep_token left out. the first sentence is Claim.
            truncate_length = max_length - 1 - len(idx_by_sentence[0])
            # truncate abstract(paragraph).
            truncated_sentences = longest_first_truncation(idx_by_sentence[1:], truncate_length)
            truncated_paragraph = torch.cat([idx_by_sentence[0]]
                                            + truncated_sentences + [torch.tensor([sep_token_id])], 0)
            all_paragraphs.append(truncated_paragraph.unsqueeze(0))
    return torch.cat(all_paragraphs, 0)


def encode_paragraph(tokenizer, claim, abstract, max_sent_len=512):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_dict = tokenizer.batch_encode_plus(
        list(zip(claim, abstract)),
        # padding=True,
        padding=True,
        add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        # Too long for the model. Truncate it
        if 'token_type_ids' in encoded_dict:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'token_type_ids': encoded_dict['token_type_ids'][:, :max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:, :max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'attention_mask': encoded_dict['attention_mask'][:, :max_sent_len]
            }
    # encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict
