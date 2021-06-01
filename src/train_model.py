import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pathlib import Path
import numpy as np

# from embedding.jointmodel import JointModelClassifier
from embedding.model import JointModelClassifier
# from embedding.AutomaticWeightedLoss import AutomaticWeightedLoss
from evaluation.evaluation_model import evaluation_joint, evaluation_abstract_retrieval
from dataset.encode import encode_paragraph
from utils import token_idx_by_sentence, get_rationale_label


def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def train_base(train_set, dev_set, args):
    # awl = AutomaticWeightedLoss(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args)
    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    """
    """
    best_f1 = 0
    best_model = model
    model.train()
    checkpoint = os.path.join(args.save, f'JointModel.model')
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
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0],
              f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0],
              f'Epoch {epoch} dev rationale score:', dev_score[1])
        # save
        # save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
        #                          + f'-abstract_f1-{int(dev_score[0]["f1"]*1e4)}'
        #                          + f'-rationale_f1-{int(dev_score[1]["f1"]*1e4)}.model')
        # torch.save(model.state_dict(), save_path)
        if (dev_score[0]['f1'] + dev_score[1]['f1']) / 2 >= best_f1:
            best_f1 = (dev_score[0]['f1'] + dev_score[1]['f1']) / 2
            best_model = model
    torch.save(best_model.state_dict(), checkpoint)
    return checkpoint
