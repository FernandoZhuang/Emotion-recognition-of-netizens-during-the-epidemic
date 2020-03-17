'''
creater: zhw
updater: zhw
created_time: 2020.3.12
updated_time: 2020.3.14
Learn from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#1-setup
'''

import torch
import torch.utils.data as tud
import transformers
import sklearn.model_selection as sms
import numpy as np
import pandas as pd

import os
import time
import datetime
import random
import multiprocessing
import functools
import my_setting.utils as utils
import DataPreprocessing as dp
import itertools


def token_encode_multiprocess(tokenizer, sentences):
    n_cores = 8
    start_time = time.time()

    with multiprocessing.Pool(n_cores) as p:
        token_encode_partial = functools.partial(tokenizer.encode, add_special_tokens=True,
                                                 max_length=int(
                                                     utils.cfg.get('HYPER_PARAMETER', 'max_sequence_length')),
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


def attention_mask(input_ids):
    '''
    记录哪些单词被mask
    :param input_ids: Token ID
    :return: list
    '''
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return attention_masks


def create_itrator_for_dataset(input_ids=None, attention_masks=None, label_arg=None):
    '''
    把input_id,att_mask,label(如果有)转换成dataloader
    :param kwargs:
    :return: dataloader
    '''
    assert input_ids and attention_masks, f'input_ids,attention_masks必须被赋值!'

    inputs, masks = torch.tensor(input_ids), torch.tensor(attention_masks)
    if label_arg == None:
        input_data = tud.TensorDataset(inputs, masks)
    else:
        labels = torch.tensor(label_arg)
        input_data = tud.TensorDataset(inputs, masks, labels)

    input_sampler = tud.SequentialSampler(input_data)

    return tud.DataLoader(input_data, sampler=input_sampler,
                          batch_size=int(utils.cfg.get('HYPER_PARAMETER', 'batch_size')), num_workers=4)


def last_three_hidden_layer(hidden_layers, labels=None):
    last_cat = torch.cat(
        (hidden_layers[-1][:, 0], hidden_layers[-2][:, 0], hidden_layers[-3][:, 0]), 1)
    linear_layer = torch.nn.Linear(768 * 3, 3).cuda()
    logits = linear_layer(last_cat)

    if labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        outputs = [loss, ]
        outputs = outputs + [torch.nn.functional.softmax(logits, -1).cuda()]
    else:
        outputs = torch.nn.Softmax(logits, -1).cuda()

    return outputs


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    train_set = dp.LabeledDataset()
    train_data = train_set.cleaned_data
    sentences, labels = train_data.content.values, train_data.sentiment.values
    labels = labels.tolist()

    config = transformers.BertConfig.from_pretrained(utils.cfg.get('PRETRAIN_MODEL', 'original_roberta_wwm_ext_path'),
                                                     num_labels=3, output_attentions=False, output_hidden_states=True)
    tokenizer = transformers.BertTokenizer.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'original_roberta_wwm_ext_path'))
    model = transformers.BertForSequenceClassification.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'original_roberta_wwm_ext_path'), config=config)
    model.cuda()

    # Tokenize
    input_ids = token_encode_multiprocess(tokenizer, sentences)
    # Attention Masks
    attention_masks = attention_mask(input_ids)

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

    # Convert to Pytorch Datatypes
    train_dataloader = create_itrator_for_dataset(input_ids=train_inputs, attention_masks=train_masks,
                                                  label_arg=train_labels)
    validation_dataloader = create_itrator_for_dataset(input_ids=validation_inputs, attention_masks=validation_masks,
                                                       label_arg=validation_labels)

    # region Optimizer and Learning schduler
    optimizer = transformers.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    train_steps = len(train_dataloader) * int(utils.cfg.get('HYPER_PARAMETER', 'epochs'))
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                             num_training_steps=train_steps)
    # endregion

    # region Train and Eval
    random.seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    np.random.seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    torch.manual_seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    torch.cuda.manual_seed_all(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))

    loss_values = []
    epochs = int(utils.cfg.get('HYPER_PARAMETER', 'epochs'))

    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            # token_type_ids论文中用于判断next sentences
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, _ = last_three_hidden_layer(outputs[2], b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # region Validation
        print("Running Validation...")
        t0 = time.time()

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                # 此处outputs index取值和train时不一样，需要注意, 因为不需要计算loss回传
                loss, logits = last_three_hidden_layer(outputs[1], b_labels)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        # endregion
    # endregion
    print("Training complete!")

    # region Save Model
    output_dir = '../Output/Robert_wwm_ext/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Saving model to %s" % output_dir)
    # endregion


def test():
    print('Predicting labels in test sentences...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = transformers.BertTokenizer.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'fine_tuned_roberta_wwm_ext_path'))
    model = transformers.BertForSequenceClassification.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'fine_tuned_roberta_wwm_ext_path'), num_labels=3, output_attentions=False)
    model.cuda()

    test_set = dp.TestDataset()
    test_data = test_set.cleaned_data
    sentences = test_data.content.values

    predcit_input_ids = token_encode_multiprocess(tokenizer, sentences)
    predict_attention_masks = attention_mask(predcit_input_ids)
    predict_dataloader = create_itrator_for_dataset(input_ids=predcit_input_ids,
                                                    attention_masks=predict_attention_masks)

    model.eval()

    predictions = []
    for batch in predict_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch
        with torch.no_grad(): outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predict_labels = []
    for i in range(len(predictions)): predict_labels.append(np.argmax(predictions[i], axis=1).flatten().tolist())
    test_set.fill_result(list(itertools.chain(*predict_labels)))  # 把多个list合并成一个list
    test_set.submit()
    print('    DONE.')


if __name__ == '__main__':
    train()

    test()
