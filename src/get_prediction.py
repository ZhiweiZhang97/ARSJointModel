import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.encode import encode_paragraph
from embedding.jointmodel import JointModelClassifier
# from embedding.jointmodel_base import JointModelClassifier
from utils import token_idx_by_sentence, remove_dummy


def get_predictions(args, input_set, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.batch_size_gpu = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args).to(device)
    # model = JointParagraphClassifier(args.model, args.hidden_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    abstract_result = []
    rationale_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=args.batch_size_gpu, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            abstract_out, rationale_out = model(encoded_dict, transformation_indices)
            abstract_result.extend(abstract_out)
            rationale_result.extend(rationale_out)

    return abstract_result, rationale_result
