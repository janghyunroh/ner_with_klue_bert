import os
import argparse
import pickle

import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_metric
from seqeval.metrics import classification_report
import torch

# huggingface tokenizer/model
from transformers import AutoModelForTokenClassification

# huggingface trainer
from transformers import Trainer
from transformers import TrainingArguments

# Customize encoder
from ner.ner_dataset import NERCollator
from ner.ner_dataset import NERDatasetPreEncoded

# WANDB를 사용하려면 아래 문장을 주석 처리
os.environ["WANDB_DISABLED"] = "true"

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_folder', required=True, help="Directory to save trained model.")
    p.add_argument('--data_fn', required=True, help="Data file name encoded by encoding.py to train the model.")

    p.add_argument('--valid_ratio', type=float, default= 0.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs_per_fold', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default= 0.2)
    p.add_argument('--max_length', type=int, default=100)

    p.add_argument('--use_kfold', action='store_true')
    p.add_argument('--n_splits', type=int, default=1)
    p.add_argument('--fold_i', type=int, default=None, help="It would be used to train with a certain fold of data due to resource limitation.")

    config = p.parse_args()

    return config


def get_pretrained_model(model_name: str, num_labels: int):
    """
    Basically, use AutoModelForTokenClassification from Huffingface.
    This function remains for future issue.
    """
    model_loader = AutoModelForTokenClassification
    return model_loader.from_pretrained(model_name, num_labels=num_labels)


def load_data(fn, use_kfold=False, n_splits=5, shuffle=True):
    """
    Load tsv data as Dataframe.
    If use_kfold is true, a new column ['fold'] will be added for indexing each fold.
    load_data라는 이름의 함수를 정의합니다. 이 함수는 네 개의 파라미터를 받습니다:
    fn: 데이터 파일의 이름 또는 경로입니다.
    use_kfold: k-fold 교차 검증을 사용할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
    n_splits: 교차 검증에서 몇 개의 fold로 나눌지를 결정하는 정수입니다. 기본값은 5입니다.
    shuffle: 데이터를 셔플할지 여부를 나타내는 부울 값입니다. 기본값은 True입니다.
    """
    # Get sentences and labels from a dataframe.
    with open(fn, "rb") as f: # fn 경로의 파일을 바이너리 읽기 모드("rb")로 엽니다. f는 파일 객체를 참조하는 변수입니다.
        dataset = pickle.load(f) # pickle 모듈을 사용하여 파일에서 객체를 역직렬화합니다. 이를 통해 저장된 데이터가 dataset 변수에 로드됩니다.
    data = pd.DataFrame(dataset.pop('data')) # dataset 딕셔너리에서 'data' 키를 제거하면서 해당 값을 가져와 pandas 데이터프레임으로 변환합니다.

    if use_kfold:
        skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=shuffle)
        # n_splits, random_state=42, shuffle 값을 사용하여 StratifiedKFold 객체를 생성합니다. 이 객체는 클래스 비율을 유지하면서 데이터를 fold로 나눕니다.
        data['fold'] = -1 # 데이터프레임에 'fold'라는 새 열을 추가하고, 모든 값을 -1로 초기화합니다.
        for n_fold, (_, v_idx) in enumerate(skf.split(data, data['sentence_class'])):
            # StratifiedKFold의 split 메소드를 사용해 생성된 인덱스를 반복하면서 각 fold의 검증 인덱스를 가져옵니다.
            data.loc[v_idx, 'fold'] = n_fold
            # 검증 인덱스(v_idx)에 해당하는 데이터의 'fold' 열 값을 현재 fold 번호(n_fold)로 설정합니다.
        data['id'] = [x for x in range(len(data))]
        # 각 데이터 포인트에 대해 유일한 ID를 생성하여 'id' 열에 할당합니다.

    return data, dataset


def split_dataset(data, use_kfold=False, n_fold=None, valid_ratio=.2, shuffle=False):
    """
    Split data into train and validation.
    Size of validation set will be determined by 'n_fold' when 'use_kfold' is True, otherwise determined by 'valid_ratio'.
    'shuffle' will affect only in case of 'use_kfold' is False.
    """
    if use_kfold == True:
        train = data[data['fold'] != n_fold]
        # data 데이터프레임에서 'fold' 열의 값이 n_fold와 다른 모든 데이터를 학습 데이터로 선택합니다.
        valid = data[data['fold'] == n_fold]
        # data 데이터프레임에서 'fold' 열의 값이 n_fold와 같은 데이터를 검증 데이터로 선택합니다.
    else:
        train, valid = train_test_split(
            data, test_size=valid_ratio, random_state=42, shuffle=shuffle, stratify=data['sentence_class'])

    train_dataset = NERDatasetPreEncoded(train['input_ids'].values, train['attention_mask'].values, train['labels'].values)
    valid_dataset = NERDatasetPreEncoded(valid['input_ids'].values, valid['attention_mask'].values, valid['labels'].values)
    # train 데이터프레임에서 필요한 열('input_ids', 'attention_mask', 'labels')을 추출하여 NERDatasetPreEncoded 클래스의 인스턴스를 생성합니다.
    # 이 클래스는 NER(Named Entity Recognition) 작업을 위한 데이터셋을 준비합니다.
    return train_dataset, valid_dataset


class compute_metrics():

    def __init__(self, index_to_label):
        self.index_to_label = index_to_label

    def __call__(self, pred):
        """
        Compute metrics use "seqeval"
        It evaluates based on Entity Level F1 score.
        """
        metric = load_metric('seqeval')

        labels = pred.label_ids
        predictions = pred.predictions.argmax(2)

        # Discard special tokens based on true_labels.
        true_predictions = [[self.index_to_label[p] for p, l in zip(
            prediction, label) if l >= 0] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.index_to_label[l] for p, l in zip(prediction, label) if l >= 0]
                    for prediction, label in zip(predictions, labels)]

        results = metric.compute(
            predictions=true_predictions, references=true_labels)
        eval_results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        print(classification_report(true_labels, true_predictions))

        return eval_results


def train_one_fold(data, n_fold, data_args, config):
    '''
    data: 전체 데이터셋입니다.
    n_fold: 현재 훈련할 fold의 번호입니다.
    data_args: 모델과 데이터 처리에 관련된 설정을 포함한 딕셔너리입니다.
    config: 훈련 과정의 구성을 담은 설정 객체입니다.
    '''
    pretrained_model_name = data_args['pretrained_model_name'].replace('/', '_')
    # data_args 딕셔너리에서 사전 훈련된 모델의 이름을 가져와서, 이름에 포함된 모든 '/' 문자를 '_'로 변경합니다.
    # 파일 시스템에서 경로 구분자로 사용되는 '/'를 파일 이름으로 사용할 수 없기 때문입니다.

    label_to_index = data_args['label_info']['label_to_index']
    index_to_label = data_args['label_info']['index_to_label']
    pad_token = data_args['pad_token']

    train_dataset, valid_dataset = split_dataset(
        data, use_kfold=config.use_kfold, n_fold=n_fold, valid_ratio=config.valid_ratio, shuffle=True)
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    # Get pretrained model and tokenizer.
    model = get_pretrained_model(
        data_args['pretrained_model_name'], len(label_to_index))
    # get_pretrained_model 함수를 사용하여 사전 훈련된 모델을 로드합니다. 이 때, 라벨의 개수에 따라 모델이 조정됩니다.

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    # 전체 배치 크기, 총 반복 횟수, 워밍업 단계 수를 계산합니다. 이는 GPU 개수와 설정을 통해 동적으로 결정됩니다.
    n_total_iterations = int(len(train_dataset) /
                             total_batch_size * config.n_epochs_per_fold)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '# of total_iters =', n_total_iterations,
        '# of warmup_iters =', n_warmup_steps,
    )

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{pretrained_model_name}.{n_fold}",
        num_train_epochs=config.n_epochs_per_fold,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs_per_fold,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=NERCollator(pad_token=pad_token,
                                  with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics(index_to_label),
    )

    trainer.train()

    fn_prefix = '.'.join([pretrained_model_name, 
                        f"{config.n_epochs_per_fold}_epochs", 
                        f"{config.max_length}_length",
                        f"{n_fold}_fold", 
                        "pth"])
    model_fn = os.path.join(config.model_folder, fn_prefix)

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': trainer.model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'pretrained_model_name': data_args['pretrained_model_name']
    }, model_fn)


def main(config):
    data, data_args = load_data(config.data_fn, use_kfold=config.use_kfold,
                     n_splits=config.n_splits, shuffle=True)

    if config.fold_i != None: # 설정에서 fold_i가 None이 아닌 경우, 즉 사용자가 특정 fold만 훈련하고자 할 때 실행됩니다.
        print(f'=== fold {config.fold_i} of {config.n_splits} training ===')
        train_one_fold(data, config.fold_i, data_args, config)
    else: # fold_i가 None인 경우, 즉 모든 fold에 대해 순차적으로 훈련하고자 할 때 실행됩니다.
        for i in range(config.n_splits): # 설정된 fold 수(config.n_splits)만큼 반복하면서 각 fold에 대한 훈련을 시작합니다.
            print(f'=== fold {i} of {config.n_splits} training ===')
            train_one_fold(data, i, data_args, config)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
