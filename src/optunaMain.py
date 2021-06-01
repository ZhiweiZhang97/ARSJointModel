import joblib
import torch
import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
import random
import optuna as opt

from dataset.loader import SciFactJointDataset, SciFactJointPredictionData
from utils import predictions2jsonl
from evaluation.evaluation_model import merge_rationale_label, evaluate_rationale_selection, evaluate_label_predictions
# from embedding.jointmodel import JointModelClassifier
from embedding.model import JointModelClassifier
from evaluation.evaluation_model import evaluation_joint, evaluation_abstract_retrieval
from dataset.encode import encode_paragraph
from utils import token_idx_by_sentence, get_rationale_label
from torch.utils.data import DataLoader
from tqdm import tqdm


def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def get_predictions(args, input_set, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.batch_size_gpu = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    # for m in model.state_dict().keys():
    #     print(m)
    # p
    abstract_result = []
    rationale_result = []
    retrieval_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=1, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            # abstract_out, rationale_out = model(encoded_dict, transformation_indices, match_indices)
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices)
            abstract_result.extend(abstract_out)
            rationale_result.extend(rationale_out)
            retrieval_result.extend(retrieval_out)

    return abstract_result, rationale_result, retrieval_result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/optuna_train_data.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/optuna_dev_data.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/optuna_dev_data.jsonl')
    parser.add_argument('--gold', type=str, default='../data/optuna_dev_data.jsonl')
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
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('-lambdas', type=int, default=[1, 2, 12])

    return parser.parse_args()


def printf(args):
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def main(trial):
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    # loader dataset
    split = True
    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/optuna_train_data.jsonl'
        claim_dev_path = '../data/optuna_dev_data.jsonl'
        claim_test_path = '../data/optuna_dev_data.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_dev_path

    # args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    # args.model = 'roberta-large'
    args.epochs = 20
    args.bert_lr = 1e-5
    args.lr = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    args.hidden_dim = 1024  # 768
    args.alpha = trial.suggest_float('alpha', 0.0, 1)
    args.lambdas[0] = trial.suggest_float('retrieval_lambda', 0.0, 1.0)
    args.lambdas[1] = trial.suggest_float('abstract_lambda', 0.0, 1.0-args.lambdas[0])
    args.lambdas[2] = trial.suggest_float('rationale_lambda', 1.0-args.lambdas[0]-args.lambdas[1], 1.0-args.lambdas[0]-args.lambdas[1])
    printf(args)
    k_train = 12
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set = SciFactJointDataset(args.corpus_path, claim_train_path, sep_token=tokenizer.sep_token, k=k_train)
    dev_set = SciFactJointDataset(args.corpus_path, claim_dev_path, sep_token=tokenizer.sep_token, k=k_train,
                                  down_sampling=False)
    test_set = SciFactJointDataset(args.corpus_path, claim_test_path,
                                   sep_token=tokenizer.sep_token, k=args.k, train=False, down_sampling=False)

    # _, test_set = train_test_split(test_set, test_size=0.1)

    # test_set = Subset(test_set, range(0, 900))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args)
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    model.train()
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
            _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
            rationale_loss *= args.lambdas[2]
            abstract_loss *= args.lambdas[1]
            sim_loss *= args.lambdas[0]
            bce_loss = args.alpha * bce_loss
            loss = abstract_loss + rationale_loss + sim_loss + bce_loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                  f' abstract loss: {round(abstract_loss.item(), 4)},'
                                  f' rationale loss: {round(rationale_loss.item(), 4)},'
                                  f' retrieval loss: {round(sim_loss.item(), 4)},'
                                  f' BCE loss: {round(bce_loss.item(), 4)}')
        scheduler.step()
        train_score = evaluation_abstract_retrieval(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train retrieval score:', train_score)
        dev_score = evaluation_abstract_retrieval(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score)
    # test_score = evaluation_abstract_retrieval(model, dev_set, args, tokenizer)
    # return test_score['abstract_recall']
    abstract_result, rationale_result, retrieval_result = get_predictions(args, test_set, model)
    rationales, labels = predictions2jsonl(test_set, abstract_result, rationale_result)
    # # merge(rationales, labels, args.merge_results)
    print('rationale selection...')
    evaluate_rationale_selection(args, "prediction/rationale_selection.jsonl")
    print('label predictions...')
    evaluate_label_predictions(args, "prediction/label_predictions.jsonl")
    print('merging predictions...')
    res = merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)
    res = res.to_dict()
    return res['sentence_selection']['f1'], res['sentence_label']['f1'], res['abstract_label_only']['f1'],\
        res['abstract_rationalized']['f1']
    # return (res['sentence_selection']['f1'] + res['sentence_label']['f1'] + res['abstract_label_only']['f1'] + res['abstract_rationalized']['f1']) / 4


if __name__ == "__main__":
    # print(80)
    study_name = 'scifact-study-Joint-Share-BioBert'  # Unique identifier of the study.
    study = opt.create_study(study_name=study_name, directions=['maximize', 'maximize', 'maximize', 'maximize'], storage='sqlite:///scifact.db')
    # study = opt.load_study('no-name-1617f854-2b37-4786-93ec-cd7662c3a6d8')
    study.optimize(main, n_trials=100)
    joblib.dump(study, "study-Joint-Share-BioBert.pkl")

    # study = joblib.load('../optuna-results/study-BioBertW.pkl')
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    # fig = opt.visualization.plot_optimization_history(study)
    # fig.show()
