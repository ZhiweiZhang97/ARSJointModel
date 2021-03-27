import jsonlines
import torch
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

NEI_LABEL = "NOT_ENOUGH_INFO"


def save_rationale_selection(output_path, results):
    output = jsonlines.open(output_path, 'w')
    # for result in results:
    #     if k:
    #         evidence = {doc_id: list(sorted(sentence_scores.argsort()[-k:][::-1].tolist()))
    #                     for doc_id, sentence_scores in result['evidence_scores'].items()}
    #     else:
    #         evidence = {doc_id: (sentence_scores >= threshold).nonzero()[0].tolist()
    #                     for doc_id, sentence_scores in result['evidence_scores'].items()}
    #     output.write({
    #         'claim_id': result['claim_id'],
    #         'evidence': evidence
    #     })
    for result in results:
        output.write({
            'claim_id': result['claim_id'],
            'evidence': result['evidence']
        })


def label_prediction(sentences, claims, args, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = {
        "claim_and_rationale": list(zip(claims, sentences)),
        "only_claim": claims,
        "only_rationale": sentences
    }[args.mode]
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        padding=True,
        return_tensors='pt'
    )
    if encoded_dict['input_ids'].size(1) > 512:
        print(encoded_dict['input_ids'].size(1))
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device)
                    for key, tensor in encoded_dict.items()}
    return encoded_dict


def save_label_predictions(output_path, results):
    output = jsonlines.open(output_path, 'w')
    for result in results:
        output.write({
            'claim_id': result['claim_id'],
            'labels': result['labels']
        })


def merge_one(rationale, label):
    """
    Merge a single rationale / label pair. Throw out NEI predictions.
    """
    evidence = rationale["evidence"]
    labels = label["labels"]
    claim_id = rationale["claim_id"]

    # Check that the documents match.
    if evidence.keys() != labels.keys():
        raise ValueError(f"Evidence docs for rationales and labels don't match for claim {claim_id}.")

    docs = sorted(evidence.keys())

    final_predictions = {}

    for this_doc in docs:
        this_evidence = evidence[this_doc]
        this_label = labels[this_doc]["label"]

        if this_label != NEI_LABEL:
            final_predictions[this_doc] = {"sentences": this_evidence,
                                           "label": this_label}

    res = {"id": claim_id,
           "evidence": final_predictions}
    return res


def merge(rationales, labels, result_file):
    """
    Merge rationales with predicted labels.
    """
    # rationales = [json.loads(line) for line in open(rationale_file)]
    # labels = [json.loads(line) for line in open(label_file)]
    # Check the ordering
    rationale_ids = [x["claim_id"] for x in rationales]
    label_ids = [x["claim_id"] for x in labels]
    if rationale_ids != label_ids:
        raise ValueError("Claim ID's for label and rationale file don't match.")

    res = [merge_one(rationale, label)
           for rationale, label in zip(rationales, labels)]

    with open(result_file, "w") as f:
        for entry in res:
            print(json.dumps(entry), file=f)


def merge_json(rationales, labels, result_file):
    labels_dict = {str(stance_json["claim_id"]): stance_json for stance_json in labels}
    results = []
    for rationale in rationales:
        id = str(rationale["claim_id"])
        result = {}
        if id in labels_dict:
            for k, v in rationale["evidence"].items():
                if len(v) > 0 and labels_dict[id]["labels"][int(k)]["label"] is not "NOT_ENOUGH_INFO":
                    result[k] = {
                        "sentences": v,
                        "label": labels_dict[id]["labels"][int(k)]["label"]
                    }
        results.append({"id": int(id), "evidence": result})
    with jsonlines.open(result_file, 'w') as output:
        for result in results:
            output.write(result)


def abstract_retrieval(dataset, output, args, state='train', include_nei=False):
    '''
    oracle-abstract
    create abstract retrieval file.
    '''
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus_path)}
    dataset = jsonlines.open(dataset)
    output = jsonlines.open(output, 'w')

    for data in dataset:
        if state == 'test':
            doc_ids = [doc_id for doc_id in corpus]
        else:
            doc_ids = list(map(int, data['evidence'].keys()))
        if not doc_ids and include_nei:
            doc_ids = [data['cited_doc_ids'][0]]

        output.write({
            'claim_id': data['id'],
            'doc_ids': doc_ids
        })


def oracle_rationale(dataset, output, args):
    '''
    oracle-rationale
    rationale selection.
    '''
    abstract = jsonlines.open(args.abstract_retrieval)
    for data, retrieval in zip(dataset, abstract):
        assert data['id'] == retrieval['claim_id']

        evidence = {}
        for doc_id in retrieval['doc_ids']:
            doc_id = str(doc_id)
            if data['evidence'].get(doc_id):
                evidence[doc_id] = [s for es in data['evidence'].get(doc_id) for s in es['sentences']]
            else:
                evidence[doc_id] = []

        output.write({
            'claim_id': data['id'],
            'evidence': evidence
        })


def oracle_tfidf_rationale(output_file, dataset, args):
    '''
    Performs sentence retrieval with oracle on SUPPORT and CONTRADICT claims,
    and tfidf on NOT_ENOUGH_INFO claims
    '''
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus_path)}
    abstract = jsonlines.open(args.abstract_retrieval)
    dataset = jsonlines.open(dataset)
    output = jsonlines.open(output_file, 'w')

    for data, retrieval in zip(dataset, abstract):
        assert data['id'] == retrieval['claim_id']

        evidence = {}

        for doc_id in retrieval['doc_ids']:
            if data['evidence'].get(str(doc_id)):
                evidence[doc_id] = [s for es in data['evidence'][str(doc_id)] for s in es['sentences']]
            else:
                sentences = corpus[doc_id]['abstract']
                vectorizer = TfidfVectorizer(stop_words='english')
                sentence_vectors = vectorizer.fit_transform(sentences)
                claim_vector = vectorizer.transform([data['claim']]).todense()
                sentence_scores = np.asarray(sentence_vectors @ claim_vector.T).squeeze()
                top_sentence_indices = sentence_scores.argsort()[-2:][::-1].tolist()
                top_sentence_indices.sort()
                evidence[doc_id] = top_sentence_indices

        output.write({
            'claim_id': data['id'],
            'evidence': evidence
        })


def tfidf_abstract(dataset, output, args, min_gram=1, max_gram=2):
    '''
    get  abstract retrieval file used tfidf.
    '''
    corpus = list(jsonlines.open(args.corpus_path))
    dataset = list(jsonlines.open(dataset))
    output = jsonlines.open(output, 'w')
    k = args.k
    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(min_gram, max_gram))
    doc_vectors = vectorizer.fit_transform([doc['title'] + ' '.join(doc['abstract'])
                                            for doc in corpus])
    for data in dataset:
        claim = data['claim']
        claim_vector = vectorizer.transform([claim]).todense()
        doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
        doc_indices_rank = doc_scores.argsort()[::-1].tolist()
        doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]
        output.write({
            'claim_id': data['id'],
            'claim': data['claim'],
            'evidence': data['evidence'] if 'evidence' in data else None,
            'cited_doc_ids': data['cited_doc_ids'] if 'cited_doc_ids' in data else None,
            'doc_ids': doc_id_rank[:k]
        })


def split_dataset(train_data):
    '''
    split train data. train data 0.8, dev data 0.2.
    '''
    claims_train_data = [claim for claim in jsonlines.open(train_data)]
    train_data, dev_data = train_test_split(claims_train_data, test_size=0.2)
    output_train = jsonlines.open('/home/g19tka09/Documents/SCIVER/data/train_data.jsonl', 'w')
    output_dev = jsonlines.open('/home/g19tka09/Documents/SCIVER/data/dev_data.jsonl', 'w')
    for data in train_data:
        output_train.write({
            'id': data['id'],
            'claim': data['claim'],
            'evidence': data['evidence'],
            'cited_doc_ids': data['cited_doc_ids'],
            'doc_ids': data['doc_ids']
        })
    for data in dev_data:
        output_dev.write({
            'id': data['id'],
            'claim': data['claim'],
            'evidence': data['evidence'],
            'cited_doc_ids': data['cited_doc_ids'],
            'doc_ids': data['doc_ids']
        })
