import torch
from torch.utils.data import Dataset

# 데이터 파일들을 모두 읽어서 학습 과정에 사용되는 데이터셋으로 
# 만들어주는 클래스들입니다.
# encoding 까지 완료된 파일들을 hf_trainer.py 에서 불러와 
# 아래 클래스들을 이용하여 batch 처리, 시퀀스 패딩, 마스킹 등을 수행합니다.

class NERCollator():
    '''
    1. batch 처리를 위한 데이터 정렬
    2. 시퀀스 패딩
    3. 텍스트, 마스크, 레이블을 텐서로 변환
    4. mini-batch 생성 시 활용됨
    '''

    def __init__(self, pad_token, with_text=True) -> None:
        self.pad_token = pad_token # ([PAD], {pad_token_id})
        self.with_text = with_text

    def __call__(self, samples):
        input_ids = [s['input_ids'] for s in samples]  # [CLS],[UNK],[SEP]가 포함되어있음
        attention_mask = [s['attention_mask'] for s in samples]
        labels = [s['labels'] for s in samples]

        # max_length 추출
        max_length = 0
        for line in input_ids:
            if max_length < len(line):
                max_length = len(line)

        # padding 추가
        for idx in range(len(input_ids)):
            # mini_batch내에 tokenize 된 문장(line)이 max_length보다 짧다면
            if len(input_ids[idx]) < max_length:
                # max_length = 원래 tokenize 된 문장(line) + ([PAD] x {max_length - len(원래 tokeniz 된 문장(line))})
                input_ids[idx] = input_ids[idx] + (
                    [self.pad_token[1]] * (max_length - len(input_ids[idx])))
                attention_mask[idx] = attention_mask[idx] + \
                    ([0] * (max_length - len(attention_mask[idx])))
                labels[idx] = labels[idx] + \
                    ([-100] * (max_length - len(labels[idx])))

        return_value = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

        return return_value


class NERDataset(Dataset):
    '''
    1. 텍스트와 레이블을 읽어옴
    2. 텍스트와 레이블을 텐서로 변환
    '''

    def __init__(self, texts, labels) -> None:
        self.texts = texts
        self.nes = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        ne = self.nes[item]

        return {
            'texts': text,
            'nes': ne,
        }


class NERDatasetPreEncoded(Dataset):

    def __init__(self, input_ids, attention_mask, labels) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        input_id = self.input_ids[item]
        attention_mask = self.attention_masks[item]
        label = self.labels[item]

        return {
            'input_ids': input_id,
            'attention_mask' : attention_mask,
            'labels': label,
        }