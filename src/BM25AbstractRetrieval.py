from gensim.summarization.bm25 import BM25
from gensim import corpora
import json
import numpy as np
import jsonlines
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="BM25 abstract retrieval"
    )
    parser.add_argument('--claim_file', type=str, default='../data/claims_test.jsonl')
    parser.add_argument('--corpus_file', type=str, default='../data/corpus.jsonl')
    parser.add_argument('--k', type=int, default=150)
    parser.add_argument('--claim_retrieved_file', type=str, default='../data/claims_test_retrieved_BM25.jsonl')

    return parser.parse_args()


def main():

    args = parse_args()

    claim_file = args.claim_file
    corpus_file = args.corpus_file

    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            abstract = json.loads(line)
            corpus[str(abstract["doc_id"])] = abstract

    claims = []
    with open(claim_file) as f:
        for line in f:
            claim = json.loads(line)
            claims.append(claim)
    claims_by_id = {claim['id']: claim for claim in claims}

    corpus_texts = []
    corpus_ids = []
    for k, v in corpus.items():
        original_sentences = [v['title']] + v['abstract']
        processed_paragraph = " ".join(original_sentences)
        corpus_texts.append(processed_paragraph)
        corpus_ids.append(k)
    corpus_ids = np.array([int(ids) for ids in corpus_ids])
    texts = [doc.split() for doc in corpus_texts]  # you can do preprocessing as removing stopwords
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = BM25(corpus)
    retrieved_corpus = {}
    for claim in claims:
        claims_doc = dictionary.doc2bow(claim['claim'].split())
        scores = bm25_obj.get_scores(claims_doc)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.k]
        retrieved_corpus[claim['id']] = corpus_ids[idx]

    with jsonlines.open(args.claim_retrieved_file, 'w') as output:
        claim_ids = sorted(list(claims_by_id.keys()))
        for id in claim_ids:
            claims_by_id[id]["doc_ids"] = retrieved_corpus[id].tolist()
            output.write(claims_by_id[id])


if __name__ == "__main__":
    main()
