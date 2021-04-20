import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pathlib import Path
import numpy as np

from embedding.jointmodel import JointModelClassifier
# from embedding.model import JointModelClassifier
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


def train(train_set, dev_set, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = JointModelClassifier(args)
    model = JointModelClassifier(args)
    model = model.to(device)

    for module in model.rationale_module:  # 冻结rationael选择
        for p in module.parameters():
            p.requires_grad = False
    for module in model.similarity_module:  # 冻结相似度层
        for p in module.parameters():
            p.requires_grad = False
    params = filter(lambda p: p.requires_grad and not model.bert.parameters(), model.parameters())
    optimizer = torch.optim.Adam([{'params': params},
                                  {'params': model.bert.parameters(), 'lr': args.lr_base}], lr=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)

    for epoch in range(args.epochs):
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            abstract_out, rationale_out, abstract_loss, rationale_loss, sim_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch["abstract_label"].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  sim_label=batch['sim_label'].to(device),)
            loss = abstract_loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0])
    #     torch.save(model.state_dict(), os.path.join(tmp_dir, f'label_module.model'))
    # model.load_state_dict(torch.load(os.path.join(tmp_dir, f'label_module.model')))

    for module in model.rationale_module:  # 解冻rationael选择
        for p in module.parameters():
            p.requires_grad = True
    for module in model.label_module:  # 冻结相似度层
        for p in module.parameters():
            p.requires_grad = False
    params = filter(lambda p: p.requires_grad and not model.bert.parameters(), model.parameters())
    optimizer = torch.optim.Adam([{'params': params},
                                  {'params': model.bert.parameters(), 'lr':  args.lr_base}], lr=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    for epoch in range(args.epochs):
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            abstract_out, rationale_out, abstract_loss, rationale_loss, sim_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch["abstract_label"].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  sim_label=batch['sim_label'].to(device),)
            loss = rationale_loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev rationale score:', dev_score[1])
    #     torch.save(model.state_dict(), os.path.join(tmp_dir, f'rationale_module.model'))
    # model.load_state_dict(torch.load(os.path.join(tmp_dir, f'rationale_module.model')))

    for module in model.similarity_module:
        for p in module.parameters():
            p.requires_grad = True
    for module in model.label_module:
        for p in module.parameters():
            p.requires_grad = True
    for module in model.rationale_module:
        for p in module.parameters():
            p.requires_grad = True
    optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': args.lr_base},
            {'params': model.sentence_attention.parameters(), 'lr': args.lr_linear},
            {'params': model.word_attention.parameters(), 'lr': args.lr_linear},
            {'params': model.rationale_linear.parameters(), 'lr': args.lr_linear},
            {'params': model.abstract_linear.parameters(), 'lr': args.lr_linear},
            {'params': model.rationale_criterion.parameters(), 'lr': args.lr_linear},
            {'params': model.abstract_criterion.parameters(), 'lr': args.lr_linear},
         ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    """
    """
    best_macro_f1 = 0
    best_f1 = 0
    best_model = model
    model.train()
    checkpoint = os.path.join(args.save, f'JointModel_SciBert.model')
    for epoch in range(args.epochs):
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            # sentence_length = torch.tensor([len for len in batch['sentence_length']])
            abstract_out, rationale_out, abstract_loss, rationale_loss, sim_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch["abstract_label"].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  sim_label=batch['sim_label'].to(device),)
                                                                  # doc_length=batch['doc_length'].to(device),
                                                                  # sentence_length=sentence_length.to(device),)
            # print('label. ', '\n', 'Truth: ', batch["abstract_label"], '\n', 'Pred: ', abstract_out)
            # print(200 * '*')
            # print('rationale. ', '\n', 'Truth: ', padded_label, '\n', 'Pred: ', rationale_out)
            # print(200 * '*')
            # print('similarity. ', '\n', 'Truth: ', batch['sim_label'], '\n', 'Pred: ', sim_out)
            # print(100 * '-*')
            rationale_loss *= 18
            abstract_loss *= 6
            # sim_loss *= 0.5
            loss = abstract_loss + rationale_loss + sim_loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                  f' abstract loss: {round(abstract_loss.item(), 4)},'
                                  f' rationale loss: {round(rationale_loss.item(), 4)},'
                                  f' similarity loss: {round(sim_loss.item(), 4)}')
        scheduler.step()
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0],
              f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0],
              f'Epoch {epoch} dev rationale score:', dev_score[1])
        # save
        # save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
        #                          + f'-abstract_f1-{int(dev_score[0]["abstract_macro_f1"]*1e4)}'
        #                          + f'-rationale_f1-{int(dev_score[1]["rationale_f1"]*1e4)}.model')
        # torch.save(model.state_dict(), save_path)
        if dev_score[0]['abstract_macro_f1'] > best_macro_f1 and dev_score[1]['rationale_f1'] > best_f1:
            best_macro_f1 = dev_score[0]['abstract_macro_f1']
            best_f1 = dev_score[1]['rationale_f1']
            best_model = model
        # checkpoint = os.path.join(args.save, f'JointModel_SciBert.model')
    torch.save(best_model.state_dict(), checkpoint)
    return checkpoint


def train_base(train_set, dev_set, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
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
    # abstract_best_f1 = 0
    # rationale_best_f1 = 0
    # best_model = model
    model.train()
    checkpoint = os.path.join(args.save, f'JointModel.model')
    # for epoch in range(10):
    #     model.train()  # cudnn RNN backward can only be called in training mode
    #     t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
    #     for i, batch in enumerate(t):
    #         encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
    #         transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
    #                                                        args.model)
    #         match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
    #                                               args.model, match=True)
    #         encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    #         transformation_indices = [tensor.to(device) for tensor in transformation_indices]
    #         match_indices = [tensor.to(device) for tensor in match_indices]
    #         _, retrieval_loss = model(encoded_dict, transformation_indices, match_indices,
    #                                   retrieval_label=batch['sim_label'].to(device), retrieval_only=True)
    #         loss = retrieval_loss
    #         loss.backward()
    #         if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             t.set_description(f'Epoch {epoch}, iter {i}, abstract retrieval loss: {round(loss.item(), 4)}')
    #     scheduler.step()
    #     train_score = evaluation_abstract_retrieval(model, train_set, args, tokenizer)
    #     print(f'Epoch {epoch} train retrieval score:', train_score)
    #     dev_score = evaluation_abstract_retrieval(model, dev_set, args, tokenizer)
    #     print(f'Epoch {epoch} dev retrieval score:', dev_score)

    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs + 10)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
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
            _, _, abstract_loss, rationale_loss, sim_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, abstract_sample=abstract_sample,
                                                                  rationale_sample=rationale_sample)

            # print('label. ', '\n', 'Truth: ', batch["abstract_label"], '\n', 'Pred: ', abstract_out)
            # print(200 * '*')
            # print('rationale. ', '\n', 'Truth: ', padded_label, '\n', 'Pred: ', rationale_out)
            # print(200 * '*')
            # print('similarity. ', '\n', 'Truth: ', batch['sim_label'], '\n', 'Pred: ', sim_out)
            # print(100 * '-*')
            rationale_loss *= 12
            abstract_loss *= 2
            sim_loss *= 1
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
        # if epoch < args.epochs // 2:
        #     train_score = evaluation_abstract_retrieval(model, train_set, args, tokenizer)
        #     print(f'Epoch {epoch} train retrieval score:', train_score)
        #     dev_score = evaluation_abstract_retrieval(model, dev_set, args, tokenizer)
        #     print(f'Epoch {epoch} dev retrieval score:', dev_score)
        # else:
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0],
              f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0],
              f'Epoch {epoch} dev rationale score:', dev_score[1])
        # save
        save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
                                 + f'-abstract_f1-{int(dev_score[0]["f1"]*1e4)}'
                                 + f'-rationale_f1-{int(dev_score[1]["f1"]*1e4)}.model')
        torch.save(model.state_dict(), save_path)
        # if dev_score[0]['abstract_f1'][1] >= abstract_best_f1 and dev_score[1]['rationale_f1'] >= rationale_best_f1:
        #     abstract_best_f1 = dev_score[0]['abstract_f1'][1]
        #     rationale_best_f1 = dev_score[1]['rationale_f1']
        #     best_model = model
        # checkpoint = os.path.join(args.save, f'JointModel_SciBert.model')
    torch.save(model.state_dict(), checkpoint)
    return checkpoint
