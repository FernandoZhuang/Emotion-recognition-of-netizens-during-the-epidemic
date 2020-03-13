'''
Learn from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#1-setup
author: zhw
time: 2020.3.12
'''

import torch
import torch.utils.data as tud
import transformers
import sklearn.model_selection as sms

import time
import multiprocessing
import functools
import Docs.utils as utils
import Code.DataPreprocessing as dp


def token_encode_multiprocess(tokenizer, sentences):
    n_cores = 8
    start_time = time.time()
    p = multiprocessing.Pool(n_cores)

    token_encode_partial = functools.partial(tokenizer.encode, add_special_tokens=True,
                                             max_length=int(utils.cfg.get('HYPER_PARAMETERS', 'max_sequence_length')),
                                             pad_to_max_length=True)
    token_encode_multiprocess_partial = functools.partial(token_encode, token_encode_partial)
    res = p.map(token_encode_multiprocess_partial, dp.Batch(n_cores, sentences.tolist()))
    res = functools.reduce(lambda x, y: x + y, res)
    print(f'已获取Token后的ID, 用时:{round(time.time() - start_time, 2)}s')

    return res


def token_encode(partial, sentences):
    '''
    若不要多进程，可基于本函数修改成单一进程tokenize，因此不使用匿名函数嵌套进token_encode_multiprocess
    '''
    return [partial(sent) for sent in sentences]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = dp.LabeledDataset()
    test_data = test_set.cleaned_data
    sentences, labels = test_data.content.values, test_data.sentiment.values

    tokenizer = transformers.BertTokenizer.from_pretrained(utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'))
    model = transformers.BertForSequenceClassification.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'), num_labels=3, output_attentions=False)
    model.cuda()

    # Tokenize
    input_ids = token_encode_multiprocess(tokenizer, sentences)
    # Attention Masks
    attention_masks = [int(token_id > 0) for sent in input_ids for token_id in sent]

    # region Train Vali Split
    train_inputs, validation_inputs, train_labels, validation_labels = sms.train_test_split(input_ids, labels,
                                                                                            random_state=2018,
                                                                                            test_size=float(
                                                                                                utils.cfg.get(
                                                                                                    'HYPER_PARAMETER',
                                                                                                    'test_size')))
    train_masks, validation_masks, _, _ = sms.train_test_split(attention_masks, labels, random_state=2018,
                                                               test_size=float(
                                                                   utils.cfg.get('HYPER_PARAMETER', 'test_size')))
    # endregion

    # region Convert to Pytorch Datatypes
    train_inputs, validation_inputs = torch.tensor(train_inputs), torch.tensor(validation_inputs)
    train_labels, validation_labels = torch.tensor(train_labels), torch.tensor(validation_labels)
    train_masks, validation_masks = torch.tensor(train_masks), torch.tensor(validation_masks)
    # endregion

    # region Create iterator for dataset
    train_data = tud.TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = tud.RandomSampler(train_data)
    train_dataloader = tud.DataLoader(train_data, sampler=train_sampler,
                                      batch_size=utils.cfg.get('HYPER_PARAMETER', 'batch_size'))

    validation_data = tud.TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = tud.RandomSampler(validation_data)
    validation_dataloader = tud.DataLoader(validation_data, sampler=validation_sampler,
                                           batch_size=utils.cfg.get('HYPER_PARAMETER', 'batch_size'))
    # endregion



    print()
