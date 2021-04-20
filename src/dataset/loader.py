import json
import os
import jsonlines
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
# from torchtext.vocab import Vocab, Vectors
import collections
import re


'''
corpus_file: {doc_id  title  abstract(sent_tokenize)  structured(True or False)}
claims_file: {id  claim(sent_tokenize)
    evidence:{
        cited_doc_ids
            sentence: ['sentenceID']
            label: SUPPORT / CONTRADICT}
    cited_doc_ids(correspond doc_id) >= 1 # max_length = 5} 
##
input of rationale selection: [claim, sentence of abstract], label=[true, false]
output: score of sentence
input of label prediction: [claim, rationale], label=[contradict, not_enough_info, support]
output: probability of label
'''


def loader_json(file_name):
    # return jsonlines.open(file_name)
    return [json.loads(line) for line in open(file_name)]


def get_corpus(corpus_path):
    return {doc['doc_id']: doc for doc in jsonlines.open(corpus_path)}


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        if 'sentence' in example:
            words += example['sentence']
        else:
            words += example['rationale']
    return words


def _sentence_to_words(text, is_abstract=False):
    if is_abstract:
        new_text = []
        for sentences in text:
            sentence2words = [word_tokenize(word) for word in sentences]
            tmp = []
            for word_id in range(len(sentence2words)):
                tmp += sentence2words[word_id]
                if word_id < len(sentence2words) - 1:
                    tmp += "\""
            new_text.append(tmp)
    else:
        new_text = [word_tokenize(word) for word in text]
    return new_text


def _data_to_array(data, is_corpus=False):
    if is_corpus:
        doc_ids = [x['doc_id'] for x in data]
        title = [x['title'] for x in data]
        abstract = [x['abstract'] for x in data]
        structured = [x['structured'] for x in data]
        new_data = {'doc_id': doc_ids,
                    'title': title,
                    'abstract': abstract,
                    'structured': structured}
    else:
        claim_id = [x['id'] for x in data]
        claims = [x['claim'] for x in data]
        evidences = [x['evidence'] for x in data]
        cited_doc_ids = [x['cited_doc_ids'] for x in data]
        new_data = {'id': claim_id,
                    'claim': claims,
                    'evidence': evidences,
                    'cited_doc_ids': cited_doc_ids}
    return new_data


def _data_encode(data, vocab, state='rationale'):
    claims = np.array([doc['claim'] for doc in data], dtype=object)
    if state == 'rationale':
        sentences = np.array([doc['sentence'] for doc in data], dtype=object)
        evidences = np.array([doc['evidence'] for doc in data])
    else:
        sentences = np.array([doc['rationale'] for doc in data], dtype=object)
        evidences = np.array([doc['label'] for doc in data])
    ############################################################################
    # compute the max text length
    claim_len = np.array([len(e) for e in claims])
    max_claim_len = max(claim_len)

    # initialize the big numpy array by <pad>
    claim_ids = vocab.stoi['<pad>'] * np.ones([len(data), min(max_claim_len, 512)],
                                              dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        claim_ids[i, :len(claims[i])] = [
            vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
            for x in claims[i]]

        # filter out document with only unk and pad
        if np.max(claim_ids[i]) < 2:
            del_idx.append(i)
    ############################################################################
    # compute the max text length
    sentence_len = np.array([len(e) for e in sentences])
    max_sentence_len = max(sentence_len)

    # initialize the big numpy array by <pad>
    sentence_ids = vocab.stoi['<pad>'] * np.ones([len(data), min(max_sentence_len, 512)],
                                                 dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        sentence_ids[i, :len(sentences[i])] = [
            vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
            for x in sentences[i][:min(len(sentences), 512)]]

        # filter out document with only unk and pad
        if np.max(claim_ids[i]) < 2:
            del_idx.append(i)

    new_data = []
    if state == 'rationale':
        for ids in range(len(claim_ids)):
            new_data.append({
                'claim_ids': claim_ids[ids],
                'claim': claims[ids],
                'claim_len': claim_len[ids],
                'sentence_ids': sentence_ids[ids],
                'sentence': sentences[ids],
                'sentence_len': sentence_len[ids],
                'evidence': evidences[ids],
            })
    else:
        for ids in range(len(claim_ids)):
            new_data.append({
                'claim_ids': claim_ids[ids],
                'claim': claims[ids],
                'claim_len': claim_len[ids],
                'rationale_ids': sentence_ids[ids],
                'rationale': sentences[ids],
                'rationale_len': sentence_len[ids],
                'label': evidences[ids],
            })
    return new_data


# def load_dataset(corpus_path, claim_path, args, state='rationale'):
#     '''
#     state: rationale / label
#     '''
#     print('Loading data')
#     if state == 'rationale':
#         all_data = SciFactRationaleSelectionDataset(corpus_path, claim_path).samples
#     elif state == 'label':
#         all_data = SciFactLabelPredictionDataset(corpus_path, claim_path).samples
#
#     print('Loading word vectors')
#     path = os.path.join(args.wv_path, args.word_vector)
#     if not os.path.exists(path):
#         # Download the word vector and save it locally:
#         print('Downloading word vectors')
#         import urllib.request
#         urllib.request.urlretrieve(
#             'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
#             path)
#
#     vectors = Vectors(args.word_vector, cache=args.wv_path)
#     vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
#                   specials=['<pad>', '<unk>'], min_freq=5)
#
#     # print word embedding statistics
#     wv_size = vocab.vectors.size()
#     print('Total num. of words: {}, word vector dimension: {}'.format(
#         wv_size[0],
#         wv_size[1]))
#
#     num_oov = wv_size[0] - torch.nonzero(
#         torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
#     print(('Num. of out-of-vocabulary words'
#            '(they are initialized to zeros): {}').format(num_oov))
#
#     new_data = _data_encode(all_data, vocab, state)
#     return new_data


class SciFactRationaleSelectionDataset(Dataset):
    '''
    create corpus-claim sentence pair with evidence.
    "evidence = True if corpus sentence is evidence of claim else False"
    '''

    def __init__(self, corpus: str, claims: str, train=True, k=3):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        # for claim in jsonlines.open(claims):
        #     for doc_id, evidence in claim['evidence'].items():
        #         doc = corpus[int(doc_id)]
        #         evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
        #         for i, sentence in enumerate(doc['abstract']):
        #             self.samples.append({
        #                 'claim': claim['claim'],
        #                 'sentence': sentence,
        #                 'evidence': i in evidence_sentence_idx
        #             })
        for claim in jsonlines.open(claims):
            if "doc_ids" in claim:
                candidates = claim["doc_ids"][:k]  # Add negative samples
            else:
                candidates = claim["cited_doc_ids"]
            candidates = [int(cand) for cand in candidates]
            evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
            all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
            if not train:
                all_candidates = candidates
            for doc_id in all_candidates:
                if str(doc_id) in claim['evidence'].keys():
                    evidence = claim['evidence'][str(doc_id)]
                else:
                    evidence = {}
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SciFactLabelPredictionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []

        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

        for claim in jsonlines.open(claims):
            if claim['evidence']:
                for doc_id, evidence_sets in claim['evidence'].items():
                    doc = corpus[int(doc_id)]

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]  # directly use the first evidence set label
                        # because currently all evidence sets have
                        # the same label
                    })

                    # Add negative samples
                    non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(2, 3), len(non_rationale_idx)))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
            else:
                # Add negative samples
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(2, 3))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SciFactAbstractRetrievalDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for claim in jsonlines.open(claims):
            cited_doc_ids = claim['cited_doc_ids']
            doc_ids = sorted(list(set(cited_doc_ids + claim['doc_ids'][:10])))
            for doc_id in doc_ids:
                doc = corpus[int(doc_id)]
                self.samples.append({
                    'claim': claim['claim'],
                    'abstract': ''.join(doc['abstract']),
                    'related': doc_id in cited_doc_ids
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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


def down_sample(abstract_sentences, evidence_sentence_idx, label, p, sep_token):
    kept_sentences = []
    evidence_map = []
    for i, sentence in enumerate(abstract_sentences):
        if random.random() < p:
            kept_sentences.append(sentence)
            if i in evidence_sentence_idx:
                evidence_map.append(True)  # i is evidence
            else:
                evidence_map.append(False)
    if len(kept_sentences) == 0:
        return None, None, None
    kept_evidence_idx = []
    for i, e in enumerate(evidence_map):
        if e:
            kept_evidence_idx.append(i)
    kept_label = label if len(kept_evidence_idx) > 0 else "NOT_ENOUGH_INFO"
    return kept_sentences, set(kept_evidence_idx), kept_label


class SciFactJointDataset(Dataset):
    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train=True, down_sampling=True):
        # sep_token = ''
        self.rationale_label = {'NOT_ENOUGH_INFO': 0, 'RATIONALE': 1}
        self.rev_rationale_label = {i: l for (l, i) in self.rationale_label.items()}
        # self.abstract_label = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}
        self.abstract_label = {'NOT_ENOUGH_INFO': 0, 'CONTRADICT': 1,  'SUPPORT': 2}
        self.rev_abstract_label = {i: l for (l, i) in self.abstract_label.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for claim in jsonlines.open(claims):
            if k > 0 and 'doc_ids' in claim:
                candidate_abstract = claim['doc_ids'][:k]
            else:
                candidate_abstract = claim['cited_doc_ids']
            candidate_abstract = [int(c) for c in candidate_abstract]
            if train:
                evidence_doc_ids = [int(id) for id in list(claim['evidence'].keys())]
                all_candidates = sorted((list(set(candidate_abstract + evidence_doc_ids))))
            else:
                all_candidates = candidate_abstract
            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                # doc_id = str(doc_id)
                abstract_sentences = [sentence.strip() for sentence in doc['abstract']]
                abstract_sentences = clean_invalid_sentence(abstract_sentences)  # #
                if train:
                    if str(doc_id) in claim['evidence']:  # cited_doc is evidence
                        evidence = claim['evidence'][str(doc_id)]
                        # print(str(doc_id), evidence)
                        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                        # print(evidence_sentence_idx)
                        labels = set(e['label'] for e in evidence)
                        if 'SUPPORT' in labels:
                            label = 'SUPPORT'
                        elif 'CONTRADICT' in labels:
                            label = 'CONTRADICT'
                        else:
                            label = 'NOT_ENOUGH_INFO'

                        if down_sampling:
                            # down samples. Augment the data set, extract sentences from the evidence abstract.
                            kept_sentences, kept_evidence_idx, kept_label = down_sample(abstract_sentences,
                                                                                        evidence_sentence_idx,
                                                                                        label, 0.5, sep_token)
                            if kept_sentences is not None:
                                concat_sentences = (' ' + sep_token + ' ').join(kept_sentences)
                                concat_sentences = clean_num(clean_url(concat_sentences))  # clean the url in the sentence
                                rationale_label_str = ''.join(
                                    ['1' if i in kept_evidence_idx else '0' for i in range(len(kept_sentences))])
                                # concat_sentences = doc['title'] + ' ' + sep_token + ' ' + concat_sentences
                                # rationale_label_str = "1" + rationale_label_str

                                self.samples.append({
                                    'claim': claim['claim'],
                                    'claim_id': claim['id'],
                                    'doc_id': doc['doc_id'],
                                    'abstract': concat_sentences,
                                    # 'paragraph': ' '.join(kept_sentences),
                                    'title': ' ' + sep_token + ' '.join(doc['title']),
                                    'sentence_label': rationale_label_str,
                                    'abstract_label': self.abstract_label[kept_label],
                                    'sim_label': 1 if doc['doc_id'] in claim['cited_doc_ids'] else 0,
                                })

                    else:  # cited doc is not evidence
                        evidence_sentence_idx = {}
                        label = 'NOT_ENOUGH_INFO'
                    concat_sentences = (' ' + sep_token +
                                        ' ').join(abstract_sentences)  # concat sentences in the abstract
                    concat_sentences = clean_num(clean_url(concat_sentences))  # clean the url in the sentence
                    rationale_label_str = ''.join(
                        ['1' if i in evidence_sentence_idx else '0' for i in range(len(abstract_sentences))])
                    # print(rationale_label_str)
                    # concat_sentences = doc['title'] + ' ' + sep_token + ' ' + concat_sentences
                    # rationale_label_str = "1" + rationale_label_str

                    self.samples.append({
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'abstract': concat_sentences,
                        # 'paragraph': ' '.join(abstract_sentences),
                        'title': ' ' + sep_token + ' '.join(doc['title']),
                        'sentence_label': rationale_label_str,
                        'abstract_label': self.abstract_label[label],
                        'sim_label': 1 if doc['doc_id'] in claim['cited_doc_ids'] else 0,
                    })
                else:
                    concat_sentences = (' ' + sep_token + ' ').join(abstract_sentences)
                    # concat_sentences = doc['title'] + ' ' + sep_token + ' ' + concat_sentences

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


class SciFactJointPredictionData(Dataset):
    def __init__(self, corpus: str, claims: str, sep_token="</s>"):
        # sep_token = ''
        self.rationale_label = {'NOT_ENOUGH_INFO': 0, 'RATIONALE': 1}
        self.rev_rationale_label = {i: l for (l, i) in self.rationale_label.items()}
        # self.abstract_label = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}
        self.abstract_label = {'NOT_ENOUGH_INFO': 0, 'CONTRADICT': 1,  'SUPPORT': 2}
        self.rev_abstract_label = {i: l for (l, i) in self.abstract_label.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


