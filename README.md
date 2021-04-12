# SciFact_JointModel
Due to the rapid growth in scientific literature, it is difficult for scientists to stay up-to-date on the latest findings. This challenge is especially acute during pandemics due to the risk of making decisions based on outdated or incomplete information. There is a need for AI systems that can help scientists with information overload and support scientific fact checking and evidence synthesis.

1. Take a scientific claim as input
2. Identify all relevant abstracts in a large corpus
3. Label them as Supporting or Refuting the claim
4. Select sentences as evidence for the label

### Dataset
* A list of abstracts from the corpus containing relevant evidence.
* A label indicating whether each abstract Supports or Refutes the claim.
* All evidence sets found in each abstract that justify the label. An evidence set is a collection of sentences that, taken together, verifies the claim. Evidence sets can be one or more sentences.

***An example of a claim paired with evidence from a single abstract is shown below.***
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
