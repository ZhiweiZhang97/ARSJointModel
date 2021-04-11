import torch
import argparse
from transformers import AutoTokenizer
import numpy as np
import random

from dataset.loader import SciFactJointDataset
from train_model import train, train_base
from get_prediction import get_predictions
from utils import predictions2jsonl
from evaluation.evaluation_model import merge_rationale_label, evaluate_rationale_selection, evaluate_label_predictions


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
                        default='../data/claims_test.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--rationale_selection_tfidf', type=str, default='prediction/rationale_selection_tfidf.jsonl')

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--rationale_model', type=str, default='')
    parser.add_argument('--label_model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--lr-base', type=float, default=1e-5)
    parser.add_argument('--lr-linear', type=float, default=5e-6)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument('--embedding', type=str, default='roberta')

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_label", type=int, default=2, help="numbers of the label")
    parser.add_argument("--class_num_label", type=int, default=1,
                        help="max number of the label for one class")
    parser.add_argument("--embed_size", type=int, default=768, help="embedding size")
    parser.add_argument("--cnn_num_filters", type=int, default=128,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")
    parser.add_argument('--vocab_size', type=int, default=31116)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="tfidf")

    return parser.parse_args()


def printf(args):
    for k in list(vars(args).keys()):
        if k == 'epochs':
            print('%s: %s' % (k, vars(args)[k]))
        if k == 'model':
            print('%s: %s' % (k, vars(args)[k]))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader dataset
    split = True
    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data.jsonl'
        claim_dev_path = '../data/dev_data.jsonl'
        claim_test_path = '../data/claims_dev_retrieved_tfidf.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_test_path

    # args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    args.model = 'dmis-lab/biobert-base-cased-v1.1-mnli'
    # args.model = 'roberta-large'
    args.epochs = 20
    args.lr_base = 1e-5
    args.lr_linear = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    args.hidden_dim = 768  # 768
    printf(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set = SciFactJointDataset(args.corpus_path, claim_train_path, sep_token=tokenizer.sep_token, k=12)
    dev_set = SciFactJointDataset(args.corpus_path, claim_dev_path, sep_token=tokenizer.sep_token, k=12)
    test_set = SciFactJointDataset(args.corpus_path, claim_test_path,
                                   sep_token=tokenizer.sep_token, k=args.k, train=False)
    # print(test_set.samples[0])
    checkpoint = train_base(train_set, dev_set, args)
    # checkpoint = 'model/JointModel.model'
    # checkpoint = 'tmp-runs/161769069367461-abstract_f1-9804-rationale_f1-6108.model'
    # checkpoint = 'tmp-runs/161778744561827-abstract_f1-9918-rationale_f1-6435.model'
    # print(checkpoint)
    abstract_result, rationale_result = get_predictions(args, test_set, checkpoint)
    rationales, labels = predictions2jsonl(test_set.samples, abstract_result, rationale_result)
    # merge(rationales, labels, args.merge_results)
    print('rationale selection...')
    evaluate_rationale_selection(args, "prediction/rationale_selection.jsonl")
    print('label predictions...')
    evaluate_label_predictions(args, "prediction/label_predictions.jsonl")
    print('merging predictions...')
    merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)


if __name__ == "__main__":
    main()
