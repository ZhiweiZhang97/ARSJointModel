import argparse
import time

import numpy as np
import random

from dataset.loader import FEVERParagraphBatchDataset
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from embedding.jointmodel import JointModelClassifier
from dataset.encode import encode_paragraph
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import token_idx_by_sentence, get_rationale_label, flatten


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/fever_train_retrieved_15.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/fever_dev_retrieved_15.jsonl')
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
    parser.add_argument('--bert-lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=5e-6)
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
    parser.add_argument('--k', type=int, default=10, help="number of abstract retrieval(training)")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lambdas', type=float, default=[1, 2, 12])

    return parser.parse_args()


def printf(args):
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def evaluation(model, dataset, args, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    abstract_preds = []
    abstract_labels = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            if padded_label.size(1) == transformation_indices[-1].size(1)-1:
                abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices)
                abstract_preds.extend(abstract_out)
                abstract_labels.extend(batch['abstract_label'])

                rationale_predictions.extend(rationale_out)
                rationale_labels.extend(rationale_label)

    stance_f1 = f1_score(abstract_labels, abstract_preds, average="micro", labels=[1, 2])
    stance_precision = precision_score(abstract_labels, abstract_preds, average="micro", labels=[1, 2])
    stance_recall = recall_score(abstract_labels, abstract_preds, average="micro", labels=[1, 2])
    rationale_f1 = f1_score(flatten(rationale_labels), flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels), flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels), flatten(rationale_predictions))
    return stance_f1, stance_precision, stance_recall, rationale_f1, rationale_precision, rationale_recall


def train_base(train_set, dev_set, args):
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = JointModelClassifier(args)
    model = JointModelClassifier(args)
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    """
    """
    model.train()
    checkpoint = os.path.join(args.save, f'PreTrainingJointModel.model')

    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs + 10)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            if padded_label.size(1) == transformation_indices[-1].size(1) - 1:
                _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
                # [0.2, 1.1, 12.0]
                rationale_loss *= 12
                abstract_loss *= 1.1
                sim_loss *= 0.2
                bce_loss *= 1.9
                loss = abstract_loss + rationale_loss + sim_loss
                loss.backward()
                if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                      f' abstract loss: {round(abstract_loss.item(), 4)},'
                                      f' rationale loss: {round(rationale_loss.item(), 4)},'
                                      f' retrieval loss: {round(sim_loss.item(), 4)}')
        scheduler.step()
        subset_train = Subset(train_set, range(0, 10000))
        train_score = evaluation(model, subset_train, args, tokenizer)
        print(
            f'Epoch {epoch}, train abstract f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % train_score)

        subset_dev = Subset(dev_set, range(0, 10000))
        dev_score = evaluation(model, subset_dev, args, tokenizer)
        print(
            f'Epoch {epoch}, train abstract f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)
        save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
                                 + f'-abstract_f1-{int(dev_score[0] * 1e4)}'
                                 + f'-rationale_f1-{int(dev_score[3] * 1e4)}.model')
        torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), checkpoint)
    return checkpoint


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    claim_train_path = args.claim_train_path
    claim_dev_path = args.claim_dev_path

    args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    args.epochs = 10
    args.bert_lr = 1e-5
    args.lr = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    args.hidden_dim = 1024  # 768
    printf(args)
    k_train = 5
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set = FEVERParagraphBatchDataset(claim_train_path, sep_token=tokenizer.sep_token, k=k_train)
    dev_set = FEVERParagraphBatchDataset(claim_dev_path, sep_token=tokenizer.sep_token, k=k_train)

    checkpoint = train_base(train_set, dev_set, args)
    model = JointModelClassifier(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.load_state_dict(torch.load(checkpoint))
    subset_dev = Subset(dev_set, range(0, 10000))
    dev_score = evaluation(model, subset_dev, args, tokenizer)
    print(f'Test abstract f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)


if __name__ == "__main__":
    main()
