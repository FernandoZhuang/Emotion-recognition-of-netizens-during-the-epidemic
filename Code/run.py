'''
Learn from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#1-setup
author: zhw
time: 2020.3.12
'''

import Docs.utils as utils
import torch
import tensorflow as tf
import transformers
import pandas as pd

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(utils.cfg.get('TSV_DATA', 'train_tsv_path'), delimiter='\t')
    sentences, labels = df.head4.values, df.head1.values

    tokenizer = transformers.BertTokenizer.from_pretrained(utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'))
    model = transformers.BertModel.from_pretrained(utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'))

    # region Tokenize
    input_ids = []
    for sent in sentences:
        encode_sent = tokenizer.encode(sent, add_special_tokens=True,
                                       max_length=int(utils.cfg.get('HYPER_PARAMETERS', 'max_sequence_length')),
                                       pad_to_max_length=True)
        input_ids.append(encode_sent)
    # endregion

    print()
