import argparse
import json
import random
import re
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader

from embedding.jointmodel import JointModelClassifier
from dataset.encode import encode_paragraph
from evaluation.data import GoldDataset, PredictedDataset
from evaluation.metrics import compute_metrics
from get_prediction import get_predictions
from evaluation.evaluation_model import merge_rationale_label
from src.dataset import loader
from src.dataset.utils import merge_retrieval
from utils import token_idx_by_sentence


def merge(args, state='valid', gold=''):
    '''
    # ================================================================================================================ #
    # merge rationale and label predictions and abstract retrieval results.
    # evaluate final predictions.
    # ================================================================================================================ #
    '''
    print('evaluate final predictions result...')

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
        return res
    else:
        print('')
        return None


class SciFactJointPredictionData(Dataset):
    def __init__(self, corpus: str, claims: str, retrieval: str,sep_token="</s>"):
        # sep_token = ''
        self.rationale_label = {'NOT_ENOUGH_INFO': 0, 'RATIONALE': 1}
        self.rev_rationale_label = {i: l for (l, i) in self.rationale_label.items()}
        # self.abstract_label = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}
        self.abstract_label = {'NOT_ENOUGH_INFO': 0, 'CONTRADICT': 1,  'SUPPORT': 2}
        self.rev_abstract_label = {i: l for (l, i) in self.abstract_label.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        abstract_retrieval = jsonlines.open(retrieval)
        for claim, retrieval in list(zip(jsonlines.open(claims), abstract_retrieval)):
            for doc_id in retrieval['doc_ids']:
                doc = corpus[int(doc_id)]
                # doc_id = str(doc_id)
                abstract_sentences = [sentence.strip() for sentence in doc['abstract']]
                abstract_sentences = clean_invalid_sentence(abstract_sentences)  # #
                concat_sentences = (' ' + sep_token + ' ').join(abstract_sentences)
                title = clean_num(clean_url(doc['title']))
                concat_sentences = title + ' ' + sep_token + ' ' + concat_sentences

                self.samples.append({
                    'claim': claim['claim'],
                    'claim_id': claim['id'],
                    'doc_id': doc['doc_id'],
                    'abstract': concat_sentences,
                    # 'paragraph': ' '.join(abstract_sentences),
                    'title': ' ' + sep_token + ' '.join(doc['title']),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def predictions2jsonl(args, claims_file, abstract_results, rationale_results):
    claim_ids = []
    claims = {}
    output_rationale = "prediction/rationale_selection.jsonl"
    output_labels = "prediction/label_predictions.jsonl"
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

    claim_ids = []
    claims = {}
    # LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']
    LABELS = ['NOT_ENOUGH_INFO', 'CONTRADICT', 'SUPPORT']
    for claim, abstract in zip(claims_file, abstract_results):
        claim_id = claim['claim_id']
        claim_ids.append(claim_id)

        curr_claim = claims.get(claim_id, {'claim_id': claim_id, 'labels': {}})
        curr_claim['labels'] = curr_claim['labels']
        curr_claim['labels'][claim['doc_id']] = {'label': LABELS[abstract], 'confidence': 1}
        claims[claim_id] = curr_claim
    label_claim = [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]

    assert (len(rationale_claim) == len(label_claim))
    for abstract, rationale in zip(label_claim, rationale_claim):
        assert (abstract["claim_id"] == rationale["claim_id"])
        for doc_id, pred in rationale["evidence"].items():
            if len(pred) == 0:
                abstract["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"

    claims = jsonlines.open(args.gold)
    for claim in claims:
        claim_id = claim['id']
        is_present = False
        for r_c in rationale_claim:
            if claim_id in r_c.values():
                is_present = True
        if not is_present:
            rationale_claim.append({'claim_id': claim_id, 'evidence': {}})
            label_claim.append({'claim_id': claim_id, 'labels': {}})
    # print(rationale_claim)
    rationale_claim.sort(key=lambda i: i.get('claim_id', 0))
    label_claim.sort(key=lambda i: i.get('claim_id', 0))

    with open(output_rationale, "w") as f:
        for entry in rationale_claim:
            print(json.dumps(entry), file=f)

    with open(output_labels, "w") as f:
        for entry in label_claim:
            print(json.dumps(entry), file=f)

    return rationale_claim, label_claim


def clean_url(word):
    """
        Clean specific data format from social media
    """
    # clean urls
    word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
    word = re.sub(r'exlink', '<URL>', word)
    return word


def clean_num(word):
    # check if the word contain number and no letters
    if any(char.isdigit() for char in word):
        try:
            num = float(word.replace(',', ''))
            return '@'
        except:
            if not any(char.isalpha() for char in word):
                return '@'
    return word


def clean_invalid_sentence(abstract):
    sentences = []
    for sen in abstract:
        sen = re.sub(r'[?\.?\s]', ' ', sen).strip()
        if sen != '':
            sentences.append(sen)
    return sentences


def get_oracle_result(args, input_set, checkpoint):
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus_path)}
    dataset = jsonlines.open(input_set)
    abstract_retrieval = jsonlines.open(args.abstract_retrieval)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    LABELS = ['NOT_ENOUGH_INFO', 'CONTRADICT', 'SUPPORT']
    result = []
    output = jsonlines.open(args.merge_results, 'w')
    for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
        assert data['id'] == retrieval['id']
        claim = data['claim']
        evidence = {}
        for doc_id in retrieval['doc_ids']:
            doc = corpus[doc_id]
            abstract_sentences = [sentence.strip() for sentence in doc['abstract']]
            abstract_sentences = clean_invalid_sentence(abstract_sentences)
            concat_sentences = (' ' + tokenizer.sep_token + ' ').join(abstract_sentences)
            title = clean_num(clean_url(doc['title']))
            abstract = title + ' ' + tokenizer.sep_token + ' ' + concat_sentences
            encoded_dict = encode_paragraph(tokenizer, [claim], [abstract])
            # print(encoded_dict)
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            abstract_out, rationale_out, _ = model(encoded_dict, transformation_indices)
            rationale_sentence = []
            for i, sen in enumerate(rationale_out[0]):
                if sen == 1:
                    rationale_sentence.append(i)
            label = LABELS[abstract_out[0]]
            if label != 'NOT_ENOUGH_INFO':
                evidence[doc_id] = {'sentences': rationale_sentence, 'label': label}
        result.append({
            'id': retrieval['id'],
            'evidence': evidence,
        })
    with open(args.merge_results, 'w') as f:
        for entry in result:
            print(json.dumps(entry), file=f)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/claims_train_retrieved.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval1.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--pre_trained_model', type=str)

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--bert-lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument('--vocab_size', type=int, default=31116)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="number of abstract retrieval(training)")
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--lambdas', type=float, default=[1, 2, 12])

    return parser.parse_args()


def printf(args, split):
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    if split:
        print('split: True')
    else:
        print('split: False')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader dataset
    split = False
    prediction = False
    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data_Bio.jsonl'
        claim_dev_path = '../data/dev_data_Bio.jsonl'
        claim_test_path = '../data/claims_dev_retrieved.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_dev_path

    # args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    # args.pre_trained_model = 'model/pre-train.model'
    args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    # args.model = 'roberta-large'
    args.epochs = 40
    args.bert_lr = 1e-5
    args.lr = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    args.hidden_dim = 1024  # 768/1024
    # args.alpha = 1.9  # BioBert-large
    args.alpha = 2.2  # RoBerta-large
    # args.lambdas = [0.2, 1.1, 12.0]  # BioBert-large w   /get
    args.lambdas = [0.9, 2.6, 11.1]  # RoBerta-large w
    # args.lambdas = [0.1, 4.7, 10.8]  # BioBert-large w/o /get
    # args.lambdas = [2.7, 2.2, 11.7]  # RoBerta-large w/o
    printf(args, split)
    # k_train = 12
    claim_test_path = 'prediction/abstract_retrieval.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    test_set = SciFactJointPredictionData(args.corpus_path, claim_test_path, args.abstract_retrieval, sep_token=tokenizer.sep_token)
    # checkpoint = 'model/BioBert_large_w.model'
    # print('BioBert-large w:')
    # abstract_result, rationale_result, retrieval_result = get_predictions(args, test_set, checkpoint)
    # rationales, labels = predictions2jsonl(args, test_set.samples, abstract_result, rationale_result)
    # get_oracle_result(args, claim_test_path, checkpoint)
    # merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)
    # merge_results = loader.loader_json(args.merge_results)
    # abstract = loader.loader_json("prediction/abstract_retrieval.jsonl")
    #
    # merge_retrieval(merge_results, abstract, 'prediction/mergeresult.jsonl')
    #
    # merge(args, state='valid', gold=args.gold)

    checkpoint = 'model/BioBert_large_w.model'
    # print('BioBert-large w/o:')
    # abstract_result, rationale_result, retrieval_result = get_predictions(args, test_set, checkpoint)
    # rationales, labels = predictions2jsonl(args, test_set.samples, abstract_result, rationale_result)
    get_oracle_result(args, claim_test_path, checkpoint)
    # merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)
    merge(args, state='valid', gold=args.gold)
    '''
    args.model = 'roberta-large'
    checkpoint = 'model/RoBerta_large_w.model'
    print('RoBerta-large w:')
    # abstract_result, rationale_result, retrieval_result = get_predictions(args, test_set, checkpoint)
    # rationales, labels = predictions2jsonl(args, test_set.samples, abstract_result, rationale_result)
    get_oracle_result(args, claim_test_path, checkpoint)
    # merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)
    merge(args, state='valid', gold=args.gold)

    checkpoint = 'model/RoBerta_large_wo.model'
    print('RoBerta-large w/o:')
    # abstract_result, rationale_result, retrieval_result = get_predictions(args, test_set, checkpoint)
    # rationales, labels = predictions2jsonl(args, test_set.samples, abstract_result, rationale_result)
    get_oracle_result(args, claim_test_path, checkpoint)
    # merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)
    merge(args, state='valid', gold=args.gold)
    '''

if __name__ == "__main__":
    main()
