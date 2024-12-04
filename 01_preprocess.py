'''
1. 데이터 전처리 코드
KLUE-BERT 모델 학습을 위해 데이터 전처리를 하는 코드입니다. 

1) docanno 툴을 이용해 태깅 완료된 json 파일을 tsv 파일로 변환합니다. 
2) 변환한 tsv 파일에서 각종 학습 과정 모니터링 및 train-test split을 위한 여러 작업을 수행합니다. 
3) 준비된 데이터셋에 대한 임베딩을 수행합니다. 
'''
# 1. 라이브러리 호출
import os
import pickle
import json
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import AdamW

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_path',
        default = './data/doccano_json',
        help="Path of file or directory to load original corpus."
    )
    p.add_argument(
        '--save_path',
        default = './data/raw',
        help="Path to save data."
    )
    p.add_argument(
        '--return_tsv',
        default = True,
        help="If not true, only pickle file will be saved."
    )

    config = p.parse_args(args=[])

    return config

def json_to_tsv(file: str):
    cols = ['sentence_id', 'sentence', 'ne'] # 데이터 프레임 칼럼 준비
    df = pd.DataFrame(columns=cols) # 데이터 프레임 생성
    id = 0

    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 공백 제거
            if line:  # 빈 줄 무시
                try:
                    data.append(json.loads(line))  # 한 줄씩 파싱
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
    
    ne = []

    for i in range(0, len(data)):
        if data[i]['text'] == None:
            continue
        else:
            df.loc[id, 'sentence'] = data[i]['text']

            # 엔티티 정보를 딕셔너리로 변환
            entity_dict = {}
            for index, (start, end, label) in enumerate(data[i]['label'], 1):
                # 문자열 슬라이싱을 사용하여 형태소 'form' 추출
                form = data[i]['text'][start:end]
                entity_dict[index] = {
                    'form': form,
                    'label': label,
                    'begin': start,
                    'end': end
                }
            # print(entity_dict)
            ne.append(entity_dict)
            id += 1
    df['ne'] = ne

    return df


def main():
    config = define_argparser()
    
    file = os.path.join(config.load_path, '대구지방법원 2007. 10. 2 선고 2007노1818 판결 [폭력행위.jsonl')

    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 공백 제거
            if line:  # 빈 줄 무시
                try:
                    data.append(json.loads(line))  # 한 줄씩 파싱
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")

print(data)
print(len(data))

if __name__ == '__main__':
    main()