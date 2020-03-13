'''
Learn from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#1-setup
author: zhw
time: 2020.3.12
'''

import torch
import torch.utils.data as tud
import transformers
import sklearn.model_selection as sms
import numpy as np

import time
import datetime
import random
import multiprocessing
import functools
import Docs.utils as utils
import Code.DataPreprocessing as dp


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


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = dp.LabeledDataset()
    test_data = test_set.cleaned_data
    sentences, labels = test_data.content.values, test_data.sentiment.values
    labels = labels.astype(np.int).tolist()

    tokenizer = transformers.BertTokenizer.from_pretrained(utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'))
    model = transformers.BertForSequenceClassification.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'roberta_wwm_ext_path'), num_labels=3, output_attentions=False)
    model.cuda()

    # Tokenize
    input_ids = token_encode_multiprocess(tokenizer, sentences)
    # Attention Masks
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

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
                                      batch_size=int(utils.cfg.get('HYPER_PARAMETER', 'batch_size')))

    validation_data = tud.TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = tud.RandomSampler(validation_data)
    validation_dataloader = tud.DataLoader(validation_data, sampler=validation_sampler,
                                           batch_size=int(utils.cfg.get('HYPER_PARAMETER', 'batch_size')))
    # endregion

    # region Optimizer and Learning schduler
    optimizer = transformers.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    train_steps = len(train_dataloader) * int(utils.cfg.get('HYPER_PARAMETER', 'epochs'))
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                             num_training_steps=train_steps)
    # endregion

    # region Train
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
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
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
                logits = outputs[0]
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
