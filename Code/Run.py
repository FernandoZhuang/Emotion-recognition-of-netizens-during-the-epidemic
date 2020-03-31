'''
Learn from
https://mccormickml.com/2019/07/22/BERT-fine-tuning/#1-setup
https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
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
import tqdm
import multiprocessing
import functools
import my_setting.utils as utils
import DataPreprocessing as dp
import Hashtag as ht
import itertools


class Dataset():
    def __init__(self, preprocessed_data=None, tokenizer=None):
        super(Dataset, self).__init__()

        self.preprocessed_data = preprocessed_data
        self.tokenizer = tokenizer

    def get_dataloader(self):
        cleaned_data = self.preprocessed_data.cleaned_data
        sentences = cleaned_data.content.values

        if self.tokenizer is None:
            tokenizer = transformers.BertTokenizer.from_pretrained(
                utils.cfg.get('PRETRAIN_MODEL', 'original_bert_path'))
        else:
            tokenizer = transformers.BertTokenizer.from_pretrained(
                utils.cfg.get('PRETRAIN_MODEL', 'fine_tuned_bert_path'))

        # TODO input_ids attention_mask要不要做私有属性 在LabeledDataset使用get方法获取?
        self.input_ids = self.token_encode_multiprocess(tokenizer=tokenizer, sentences=sentences)
        self.attention_masks = self.attention_mask(self.input_ids)

        return self.create_itrator_for_dataset(self.input_ids, self.attention_masks)

    def token_encode(self, partial, sentences):
        '''
        若不要多进程，可基于本函数修改成单一进程tokenize，因此不使用匿名函数嵌套进token_encode_multiprocess
        '''
        return [partial(sent) for sent in sentences]

    def token_encode_multiprocess(self, tokenizer, sentences):
        n_cores = 10
        start_time = time.time()

        with multiprocessing.Pool(n_cores) as p:
            token_encode_partial = functools.partial(tokenizer.encode, add_special_tokens=True,
                                                     max_length=int(
                                                         utils.cfg.get('HYPER_PARAMETER', 'max_sequence_length')),
                                                     pad_to_max_length=True)
            token_encode_multiprocess_partial = functools.partial(self.token_encode, token_encode_partial)
            res = p.map(token_encode_multiprocess_partial, dp.Batch(n_cores, sentences.tolist()))
            res = functools.reduce(lambda x, y: x + y, res)
            print(f'已获取Token后的ID, 用时:{round(time.time() - start_time, 2)}s')

        return res

    def attention_mask(self, input_ids):
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

    def create_itrator_for_dataset(self, input_ids=None, attention_masks=None, label_arg=None):
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


class LabeledDataset(Dataset):
    def __init__(self, preprocessed_data=None, tokenizer=None):
        super(LabeledDataset, self).__init__(preprocessed_data, tokenizer)

    def get_dataloader(self):
        # 把input_ids attention_masks 赋值
        super().get_dataloader()
        # TODO labels要不要做私有属性?
        labels = self.preprocessed_data.cleaned_data.sentiment.values
        labels = labels.tolist()

        train_inputs, validation_inputs, train_labels, validation_labels = sms.train_test_split(self.input_ids, labels,
                                                                                                random_state=2018,
                                                                                                test_size=float(
                                                                                                    utils.cfg.get(
                                                                                                        'HYPER_PARAMETER',
                                                                                                        'test_size')))
        train_masks, validation_masks, _, _ = sms.train_test_split(self.attention_masks, labels, random_state=2018,
                                                                   test_size=float(
                                                                       utils.cfg.get('HYPER_PARAMETER', 'test_size')))

        train_dataloader = self.create_itrator_for_dataset(train_inputs, train_masks, train_labels)
        validation_dataloader = self.create_itrator_for_dataset(validation_inputs, validation_masks, validation_labels)

        return train_dataloader, validation_dataloader


class BertForSeqClassification(torch.nn.Module):
    def __init__(self, hidden_layers=None, pool_out=None, labels=3):
        '''
        :param hidden_layers:
        :param pool_out:
        :param labels:
        '''
        super(BertForSeqClassification, self).__init__()
        self._hidden_size = 768
        self.hidden_layers, self.pool_out, self.labels = hidden_layers, pool_out, labels

        self._config = transformers.BertConfig.from_pretrained(
            utils.cfg.get('PRETRAIN_MODEL', 'original_bert_path'), num_labels=self.labels,
            output_attentions=False,
            output_hidden_states=True)
        self.bert = transformers.BertForSequenceClassification.from_pretrained(
            utils.cfg.get('PRETRAIN_MODEL', 'original_bert_path'), config=self._config)

        self.dropout = torch.nn.Dropout(float(utils.cfg.get('HYPER_PARAMETER', 'hidden_dropout_prob')))
        self.loss = torch.nn.CrossEntropyLoss()
        if self.hidden_layers is not None: torch.nn.init.xavier_normal_(self.classifier.weight)

    @property
    def classifier(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.hidden_layers is None:
            return None
        elif self.pool_out is None:
            return torch.nn.Linear(self._hidden_size * self.hidden_layers, self.labels).to(device)
        else:
            return torch.nn.Linear(self._hidden_size * self.hidden_layers + self.labels, self.labels).to(device)

    def forward(self, b_input_ids, attention_mask, label_vec=None):
        outputs = self.bert(b_input_ids, token_type_ids=None, attention_mask=attention_mask, labels=label_vec)

        if self.hidden_layers is not None:
            outputs = self._concatenate_hidden_layer_pool_out(outputs, label_vec)

        return outputs

    def _concatenate_hidden_layer_pool_out(self, original_output, label_vec):
        cat_seq = (original_output[-2],) if self.pool_out is not None else ()
        # 之所以用-1索引，应对train, vali不同情景下origianl_output是否含有loss成员，导致hidder_layer索引可变
        cat_seq = cat_seq + tuple(original_output[-1][-(i + 1)][:, 0] for i in range(self.hidden_layers))
        last_cat = torch.cat(cat_seq, 1)
        logits = self.classifier(last_cat)

        if label_vec is not None:
            loss = self.loss(logits.view(-1, self.labels), label_vec.view(-1))
            outputs = [loss, ]
            outputs = outputs + [torch.nn.functional.softmax(logits, -1)]
            # outputs = outputs + [logits]
        else:
            outputs = [torch.nn.functional.softmax(logits, -1)]
            # outputs=[logits]

        return outputs


def train():
    '''
    默认fine-tune后，紧接着预测。可注释，从本地加载再预测
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed()

    ld = LabeledDataset(preprocessed_data=dp.LabeledDataset(), tokenizer=None)
    # 如果要拼接隐藏层和pool out，此处实例化需要相应传参数
    # model = BertForSeqClassification(hidden_layers=2, pool_out=True, labels=3).to(device)
    model = BertForSeqClassification(labels=3).to(device)
    loss_values = []

    train_dataloader, validation_dataloader = ld.get_dataloader()
    epochs = int(utils.cfg.get('HYPER_PARAMETER', 'epochs'))
    train_steps = len(train_dataloader) * epochs

    # region Optimizer and Learning schduler
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': float(utils.cfg.get('HYPER_PARAMETER', 'weight_decay'))},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        utils.cfg.get('HYPER_PARAMETER', 'warmup_steps')), num_training_steps=train_steps)
    # endregion

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
            outputs = model(b_input_ids, attention_mask=b_input_mask, label_vec=b_labels)
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
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        # endregion

    print("Training complete!")

    # region Save Model
    output_dir = '../Output/Bert_base_Chinese/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    # 自定义模型无save_pretrained方法
    # model_to_save.save_pretrained(output_dir)
    output_file = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), output_file)

    print("Saving model to %s" % output_dir)
    # endregion

    # Test
    test(model)


def test(model=None):
    print('Predicting labels in test sentences...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hashtag = ht.Hashtag()

    if model is None:
        model = BertForSeqClassification()
        model.load_state_dict(
            torch.load((utils.cfg.get('PRETRAIN_MODEL', 'fine_tuned_bert_path') + '/pytorch_model.bin')))
        model.to(device)

        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    tokenizer = transformers.BertTokenizer.from_pretrained(
        utils.cfg.get('PRETRAIN_MODEL', 'fine_tuned_bert_path'))
    model.eval()

    test_set = dp.TestDataset()
    ul = Dataset(test_set, tokenizer)
    predict_dataloader = ul.get_dataloader()

    predictions = []
    for batch in tqdm.tqdm(predict_dataloader):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch
        with torch.no_grad(): outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    # bayes
    predictions = hashtag.bayes(predictions)

    predict_labels = []
    for i in range(len(predictions)): predict_labels.append(np.argmax(predictions[i], axis=1).flatten().tolist())
    test_set.fill_result(list(itertools.chain(*predict_labels)))  # 把多个list合并成一个list
    test_set.submit()
    print('    DONE.')


def seed():
    random.seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    np.random.seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    torch.manual_seed(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))
    torch.cuda.manual_seed_all(int(utils.cfg.get('HYPER_PARAMETER', 'seed')))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    # train()

    test()
