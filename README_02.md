# 자세한 코드 

자세한 데이터셋 설명 및 조작법에 대한 문서입니다. 

## 1. 데이터 흐름

**1. 데이터 준비**: 판결문, 청구전 조사서 등의 법적 문서를 사전 정의된 형식의 텍스트 파일로 변경합니다. 이 과정에서 OCR이 수행되며, 수작업을 통해 검수됩니다. 
**2. 데이터 라벨링**: 사전 정의된 개체명 목록과 라벨링 가이드라인에 따라, docanno 툴을 이용해 데이터 라벨링을 실시합니다. 라벨링 결과는 json 파일 형식으로 저장됩니다. 
**3. 데이터 전처리**: 위 json 파일들은 data/json에 저장됩니다. 해당 파일들을 불러와 BERT의 train 및 test 입력으로 가능한 형태로 전처리합니다. 

- **json_to_tsv (data/json -> data/raw)**: json 파일들을 모두 병합하여 하나의 pickle(tsv도 추가 저장 가능)의 형태로 data/raw에 저장합니다. 데이터의 칼럼은 sentence_id, sentence, ne로 이루어져 있으며, ne는 딕셔너리의 형태로, 개체명 종류와 시작 인덱스, 끝 인덱스 등의 정보가 담겨 있습니다. 
- **preprocess (data/raw -> data/dataset)**: pickle 파일을 가져와 학습 과정 모니터링에 필요한 여러 전처리 작업을 수행합니다. 작업이 완료되면 data/dataset 경로에 마찬가지로 pickle 및 tsv 파일 형식으로 저장합니다. 
    + 가. **sentence_id 칼럼을 제거**합니다. 기존 코드에서는 sentence_id에서 데이터 출처를 알아낼 수 있는 데이터를 다루고 있었지만, 해당 프로젝트에선 sentence_id가 존재하지 않으므로 쓸모없는 칼럼입니다. 
    + 나. sentence가 존재하지 않는 **빈 문장을 제거**합니다. 데이터 OCR 및 라벨링 단계에서 이같은 문장은 제거하도록 하지만, 남아있는 문장이 있을 수 있기 때문에 별도로 제거합니다. 
- **encoding (data/dataset -> data/encoded)**: 전처리된 train과 test pickle 파일들을 불러와 BERT 입력이 가능하도록 token/positional/sentence embedding을 수행합니다. 

**4. 모델 학습**: trainer.py 파일을 통해 학습 및 테스트를 수행합니다. k-fold 여부 및 k의 값을 옵션으로 넘겨줄 수 있으며, 테스트 결과와 
**5. 모델 작동 확인**

## 2. 파일 실행 순서

아래 순서는 data/json 경로에 docanno를 이용한 labeled data들이 적절히 준비되었다는 가정 하에 수행됩니다. 

- tsv 변환
```
python 01_json_to_tsv.py
```
- 기타 전처리
```
python 02_preprocess.py
```
- 위치 임베딩, 문장 임베딩 적용
```
python 03_encoding.py
```
- 모델 학습 및 결과 모니터링

```
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle # K-Fold 미사용
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --use_kfold --n_splits 5 # K-Fold 사용
```
- 모델 사용
```
streamlit run demo_02.py
```

