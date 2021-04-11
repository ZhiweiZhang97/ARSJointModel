import torch
import torch.nn as nn
from jsonlines import jsonlines
import numpy as np


def token_idx_by_sentence(input_ids, sep_token_id, model_name, match=False):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens, 1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0), -1)
    sep_indices = torch.split(indices[sep_tokens], paragraph_lens)
    if match:
        sep_indices = (torch.tensor(np.array(sep_indices[0])[[0, -1]].tolist()),)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        if "large" in model_name and ('biobert' not in model_name):
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    # pad_sequence: stacks a list of Tensors along a new dimension, and pads them to equal length.
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence, paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    # if match:
    #     padd2maxlen = torch.tensor([[-1 for _ in range(indices_by_batch.shape[2], 512)] for _ in range(2)]).unsqueeze(0)
    #     indices_by_batch = torch.cat([indices_by_batch, padd2maxlen], 2)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,
                                                                                        indices_by_batch.size(1),
                                                                                        indices_by_batch.size(-1))
    mask = (indices_by_batch >= 0)

    return batch_indices.long(), indices_by_batch.long(), mask.long()


def get_rationale_label(labels, padding_idx=2):
    '''
    "padding label"
    :param labels: sentence labels
    :param padding_idx: padding id
    :return: label after padding and original label
    '''
    max_label_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_label_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        for j, e in enumerate(label):
            label_matrix[i, j] = int(e)
        label_list.append([int(e) for e in label])
    return label_matrix.long(), label_list


def flatten(array):
    fla = []
    for arr in array:
        fla.extend(arr)
    return fla


def predictions2jsonl(claims_file, abstract_results, rationale_results):
    claim_ids = []
    claims = {}
    output_rationale = jsonlines.open("prediction/rationale_selection.jsonl", 'w')
    output_labels = jsonlines.open("prediction/label_predictions.jsonl", 'w')
    assert (len(claims_file) == len(abstract_results))
    assert (len(claims_file) == len(rationale_results))

    for claim, rationale in zip(claims_file, rationale_results):
        claim_id = claim['claim_id']
        claim_ids.append(claim_id)

        rationale_sentence = []
        for i, sen in enumerate(rationale):
            if sen == 1:
                rationale_sentence.append(i)
        curr_claim = claims.get(claim_id, {'claim_id': claim_id, 'evidence': {}})
        curr_claim['evidence'][claim['doc_id']] = rationale_sentence
        claims[claim_id] = curr_claim
    rationale_claim = [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]
    for rationale in rationale_claim:
        output_rationale.write({
            'claim_id': rationale['claim_id'],
            'evidence': rationale['evidence']
        })

    claim_ids = []
    claims = {}
    # LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']
    LABELS = ['NOT_ENOUGH_INFO', 'CONTRADICT', 'SUPPORT']
    for claim, abstract in zip(claims_file, abstract_results):
        claim_id = claim['claim_id']
        claim_ids.append(claim_id)

        curr_claim = claims.get(claim_id, {'claim_id': claim_id, 'labels': {}})
        curr_claim['labels'][claim['doc_id']] = {'label': LABELS[abstract], 'confidence': 1}
        claims[claim_id] = curr_claim
    label_claim = [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]

    assert (len(rationale_claim) == len(label_claim))
    for abstract, rationale in zip(label_claim, rationale_claim):
        assert (abstract["claim_id"] == rationale["claim_id"])
        for doc_id, pred in rationale["evidence"].items():
            if len(pred) == 0:
                abstract["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"

    for label in label_claim:
        output_labels.write({
            'claim_id': label['claim_id'],
            'labels': label['labels']
        })

    return rationale_claim, label_claim


def remove_dummy(rationale_out):
    return [out[1:] for out in rationale_out]


