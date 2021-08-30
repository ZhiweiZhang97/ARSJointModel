# SciFact_JointModel
This is implementation of EMNLP2021 paper Abstract, Rationale, Stance: A Joint Model for Scientific Claim Verification. This work is for Shared Tasks of Scholarly Document Processing(@ NAACL 2021) -- **[SCIVER: Verifying Scientific Claims with Evidence](https://github.com/allenai/scifact)**
## 

<!-- Due to the rapid growth in scientific literature, it is difficult for scientists to stay up-to-date on the latest findings. This challenge is especially acute during pandemics due to the risk of making decisions based on outdated or incomplete information. There is a need for AI systems that can help scientists with information overload and support scientific fact checking and evidence synthesis.

1. Take a scientific claim as input
2. Identify all relevant abstracts in a large corpus
3. Label them as Supporting or Refuting the claim
4. Select sentences as evidence for the label -->

## Dataset
* A list of abstracts from the corpus containing relevant evidence.
* A label indicating whether each abstract Supports or Refutes the claim.
* All evidence sets found in each abstract that justify the label. An evidence set is a collection of sentences that, taken together, verifies the claim. Evidence sets can be one or more sentences.

*An example of a claim paired with evidence from a single abstract is shown below.*
``` python
{
  "id": 52,
  "claim": "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
  "evidence": {
    "11": [                     # 2 evidence sets in document 11 support the claim.
       {"sentences": [0, 1],    # Sentences 0 and 1, taken together, support the claim.
        "label": "SUPPORT"},
       {"sentences": [11],      # Sentence 11, on its own, supports the claim.
        "label": "SUPPORT"}
    ],
    "15": [                     # A single evidence set in document 15 supports the claim.
       {"sentences": [4], 
        "label": "SUPPORT"}
    ]
  },
  "cited_doc_ids": [11, 15]
}
```
## Evaluation
Two evaluation metrics will be used. 

**Abstract-level evaluation**

Abstract-level evaluation is similar to the FEVER score, described in the FEVER paper (Thorne et al., 2018). A predicted abstract is Correct if:

1. The predicted abstract is a relevant abstract.
2. The abstract's predicted Support or Refute label matches its gold label.
3. The abstract's predicted evidence sentences contain at least one full gold evidence set. Inspired by the FEVER score, the number of predicted sentences is limited to 3.

We then compute the F1 score over all predicted abstracts.

**Sentence-level evaluation**

Sentence-level evaluation scores the correctness of the individual predicted evidence sentences. A predicted sentence Correct if:

1. The abstract containing the sentence is labeled correctly as Support or Refute.
2. The sentence is part of some gold evidence set.
3. All other sentences in that same gold evidence set are also identified by the model as evidence sentences.

We then compute the F1 score over all predicted evidence sentences. Here's a simple [step-by-step](https://github.com/allenai/scifact/blob/master/doc/evaluation.md) example showing how these metrics are calculated.



## Dependencies

We recommend you create an anaconda environment:
``` python
conda create --name scifact python=3.7 conda-build
```
Then, from the `scifact` project root, run
``` python
conda develop .
```
Then, install Python requirements:
``` python
pip install -r requirements.txt
```
If you encounter any installation problem regarding sent2vec, please check their [repo](https://github.com/epfml/sent2vec). The BioSentVec model is available [here](https://github.com/ncbi-nlp/BioSentVec#biosentvec).
The checkpoint of Paragraph-Joint model used for the paper (trained on training set) is available here ([RoBERTa-large](https://drive.google.com/file/d/1iV_5rNC1ZYDRp-tCRoiA70YmW_OVA1Qe/view?usp=sharing)、[w/o RR RoBERTa-large](https://drive.google.com/file/d/1fQPWoXjb5mHx8aioDrqOJdP-ym11Nw8j/view?usp=sharing)、[BioBERT-large](https://drive.google.com/file/d/1O7jOkMN-jZOsWQZEQ97O6b-TBqhW3gQn/view?usp=sharing)、[w/o RR BioBERT-large](https://drive.google.com/file/d/1lMv_PBwzLspCTrriwOZyJUvkOhI4a2uA/view?usp=sharing)).


## Training of SCIFACT-JointModel
Run ```python main.py``` to training or prediction.
