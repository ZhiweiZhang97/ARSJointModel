from collections import Counter
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.loader as loader
from dataset.utils import merge, merge_json
from evaluation.metrics import compute_f1, compute_metrics
from evaluation.data import GoldDataset, PredictedDataset
from dataset.encode import encode_sen_pair, encode_sentence, encode_paragraph
from utils import token_idx_by_sentence, get_rationale_label, flatten, remove_dummy


def is_correct(pred_sentence, pred_sentences, gold_sets):
    """
    A predicted sentence is correctly identified if it is part of a gold
    rationale, and all other sentences in the gold rationale are also
    predicted rationale sentences.
    """
    for gold_set in gold_sets:
        gold_sents = gold_set["sentences"]
        if pred_sentence in gold_sents:
            if all([x in pred_sentences for x in gold_sents]):
                return True
            else:
                return False

    return False


def evaluate_rationale_selection(args, rationale_results):
    '''
    # ================================================================================================================ #
    # evaluate rationale selection results.
    # ================================================================================================================ #
    '''
    evaluation_set = args.claim_dev_path
    dataset = loader.loader_json(evaluation_set)
    rationale_results = loader.loader_json(rationale_results)
    counts = Counter()
    for data, retrieval in zip(dataset, rationale_results):
        assert data['id'] == retrieval['claim_id']

        # Count all the gold evidence sentences.
        for doc_key, gold_rationales in data["evidence"].items():
            for entry in gold_rationales:
                counts["relevant"] += len(entry["sentences"])

        for doc_id, pred_sentences in retrieval['evidence'].items():
            true_evidence_sets = data['evidence'].get(doc_id) or []

            for pred_sentence in pred_sentences:
                counts["retrieved"] += 1
                if is_correct(pred_sentence, pred_sentences, true_evidence_sets):
                    counts["correct"] += 1

    rationale_metrics = compute_f1(counts)
    print(f'F1:                {round(rationale_metrics["f1"], 4)}')
    print(f'Precision:         {round(rationale_metrics["precision"], 4)}')
    print(f'Recall:            {round(rationale_metrics["recall"], 4)}')
    print()


def evaluate_label_predictions(args, label_results):
    '''
    # ================================================================================================================ #
    # evaluate label predictions results.
    # ================================================================================================================ #
    '''
    evaluation_set = args.claim_dev_path
    # evaluation
    corpus = loader.get_corpus(args.corpus_path)
    dataset = loader.loader_json(evaluation_set)
    label_results = loader.loader_json(label_results)
    pred_labels = []
    true_labels = []

    # LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}
    LABELS = {'NOT_ENOUGH_INFO': 0, 'CONTRADICT': 1, 'SUPPORT': 2}

    for data, prediction in zip(dataset, label_results):
        assert data['id'] == prediction['claim_id']

        if args.filter:
            prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                    if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
        if not prediction['labels']:
            continue

        # claim_id = data['id']
        for doc_id, pred in prediction['labels'].items():
            pred_label = pred['label']
            true_label = {es['label'] for es in data['evidence'].get(doc_id) or []}
            assert len(true_label) <= 1, 'Currently support only one label per doc'
            true_label = next(iter(true_label)) if true_label else 'NOT_ENOUGH_INFO'
            pred_labels.append(LABELS[pred_label])
            true_labels.append(LABELS[true_label])
    # sentence_labels = [0, 1, 2] if include_nei else [0, 2]
    print(
        f'Accuracy           '
        f'{round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)}')
    print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
    print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
    print()
    print('                   [N      C      S     ]')  # C: CONTRADICT; N: NOT_ENOUGH_INFO; S: SUPPORT
    # print('                   [C      S     ]')
    print(f'F1:                {f1_score(true_labels, pred_labels, average=None, labels=[0, 1, 2]).round(4)}')
    print(f'Precision:         {precision_score(true_labels, pred_labels, average=None, labels=[0, 1, 2]).round(4)}')
    print(f'Recall:            {recall_score(true_labels, pred_labels, average=None, labels=[0, 1, 2]).round(4)}')
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, pred_labels))
    print()


def merge_rationale_label(rationale_results, label_results, args, state='valid', gold=''):
    '''
    # ================================================================================================================ #
    # merge rationale and label predictions.
    # evaluate final predictions.
    # ================================================================================================================ #
    '''
    print('evaluate final predictions result...')
    merge(rationale_results, label_results, args.merge_results)
    # merge_json(rationale_results, label_results, args.merge_results)

    if state == 'valid':
        import pandas as pd
        import numpy as np
        np.set_printoptions(threshold=np.inf)

        pd.set_option('display.width', 300)  # 设置字符显示宽度
        pd.set_option('display.max_rows', None)  # 设置显示最大行
        pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列

        data = GoldDataset(args.corpus_path, gold)
        predictions = PredictedDataset(data, args.merge_results)
        res = compute_metrics(predictions)
        if args.output is not None:
            with open(args.output, "w") as f:
                json.dump(res.to_dict(), f, indent=2)
        print(res)
    else:
        print('')


def evaluation_joint(model, dataset, args, tokenizer, mode='rationale&label'):
    model.eval()
    abstract_targets = []
    rationale_targets = []
    abstract_outputs = []
    rationale_output = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size_gpu, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            abstract_out, rationale_out = model(encoded_dict, transformation_indices)
            # abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
            # rationale_out = torch.argmax(rationale_score.cpu(), dim=-1).detach().numpy().tolist()

            abstract_targets.extend(batch['abstract_label'])
            abstract_outputs.extend(abstract_out)

            rationale_targets.extend(rationale_label)
            rationale_output.extend(rationale_out)
    if mode == 'label':
        return {
            'f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'p': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'r': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
        }
    elif mode == 'rationale':
        return {
            'f1': f1_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'p': precision_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'r': recall_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0)
        }
    else:
        return {
            'f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            # 'abstract_f1': tuple(f1_score(abstract_targets, abstract_outputs, zero_division=0, average=None)),
            'p': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'r': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
        }, {
            'f1': f1_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'p': precision_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'r': recall_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0)
        }


def evaluation_abstract_retrieval(model, dataset, args, tokenizer):
    model.eval()
    abstract_targets = []
    abstract_outputs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size_gpu, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            retrieval_out, _ = model(encoded_dict, transformation_indices,
                                     retrieval_label=batch['sim_label'].to(device), retrieval_only=True)
            # abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
            # rationale_out = torch.argmax(rationale_score.cpu(), dim=-1).detach().numpy().tolist()

            abstract_targets.extend(batch['sim_label'])
            abstract_outputs.extend(retrieval_out)
    return {
            'abstract_micro_f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
            'abstract_f1': tuple(f1_score(abstract_targets, abstract_outputs, zero_division=0, average=None)),
            'abstract_precision': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
            'abstract_recall': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
        }
