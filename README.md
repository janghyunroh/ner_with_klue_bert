# Pre-trained KLUE-BERT 모델 기반 NER 시스템 Fine-tuning

**원본 저장소**: [GitHub](https://github.com/ai2-ner-project/pytorch-ko-ner)

한국어 PLM fine-tuning 코드를 활용하여 법적 문서라는 특수 도메인에 대한 개체명 인식기를 구현했습니다. 
HugggingFace 라이브러리를 이용해 KLUE-BERT 모델을 Fine-tuning 하였으며, 보안을 위해 직접 데이터 레이블링과 데이터셋 구축을 진행했습니다. 

기존 코드를 활용하기 위해 학습 시 여러 번의 전처리를 거치게 됩니다. 


## DataSet
- 대한민국 법원의 공식 판결문과 경찰청의 청구전조사서 등 법적 문서를 원본 데이터로 사용하였습니다.
- 레이블의 경우 


## Pre-Requisite
- python 3.8 기준으로 테스트
- 설치 모듈 상세정보는 requirements.txt 파일 참고 

```bash
pip install -r requirements.txt
```


## How to Use

원본 코드
https://github.com/ai2-ner-project/pytorch-ko-ner
https://github.com/sim-so/pytorch-ko-ner-v2

가상환경 세팅
ner-v2.yaml로 가상환경 만들고 activate

데이터 변환 과정
json_to_tsv_01.ipynb (json → raw)
preprocess_00.ipynb (raw → dataset)
encoding_03.ipynb (dataset → encoded)

<train>
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --use_kfold --n_splits 5

<demo>
streamlit run demo_02.py

### Preparation
1. 여러 개의 json 파일로 저장된 데이터를 표 형식으로 변환하고 pickle로 저장합니다. pickle 파일 저장 시 이름은 load_path 경로에서 마지막 이름을 사용합니다.
```bash
python ./preprocess/json_to_tsv.py --load_path data/json/19_150tags_NamedEntity/NXNE2102008030.json --save_path data/raw
python ./preprocess/json_to_tsv.py --load_path data/json/21_150tags_NamedEntity --save_path data/raw
```
- '--load_fn' 다음에 특정 json 파일 또는 json 파일이 들어있는 폴더의 경로를 입력할 수 있습니다.
- '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
    
2. 학습과 평가 데이터셋을 만들어 train.pickle, test.pickle로 저장합니다.
데이터를 합치고 필요 없는 열, 문장이 누락된 행을 제거하는 과정이 포함됩니다.
```bash
python ./preprocess/preprocess.py --load_path data/raw --save_path data/dataset --test_size 0.15 --test_o_size 0.2 --save_all --return_tsv
```
```
4 files found :  ['SXNE21.pickle', 'SXNE2102007240.pickle', 'NXNE2102008030.pickle', 'NXNE2102203310.pickle']
file 0 :  351568
file 1 :  223962
file 2 :  150082
file 3 :  78170
|data before preprocessing| 803782
|data after preprocessing| 780546 / before dropping O sentences
|train| 663464 / |test| 117082 / before dropping O sentences
|train| 303028 / |test| 66845 / after dropping and sampling O sentences
```

- '--save_all'를 추가하면 분할하지 않은 전체 데이터도 파일로 추가로 저장합니다. 파일 이름은 data.pickle입니다.
- '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
- '--pass_drop_o'를 추가하면 NE를 포함하지 않는 문장도 데이터셋에 포함합니다. 기본적으로는 포함하지 않습니다.
- '--test_o_size'에서 정한 비율에 따라 평가 데이터에 NE를 포함하지 않는 문장을 추가합니다. 0.2를 입력하면 평가 데이터의 20%가 NE를 포함하지 않는 문장이 됩니다.
  NE를 포함하지 않는 문장을 뽑을 때 가능한 구어와 문어 문장의 비율을 반반으로 하되, 그렇게 하지 못할 경우 부족한 부분을 구어 문장으로 채웁니다.

3. 데이터셋을 선택한 모델의 토크나이저로 인코딩 합니다. 인코딩된 파일은 {원본 파일명}.{모델 이름}.encoded.pickle 이름으로 저장합니다.
```bash
python ./preprocess/encoding.py --load_fn data/dataset/train.pickle --save_path data/encoded
```
```
Tokenizer loaded : klue/roberta-base
Sentences encoded : |input_ids| 66845, |attention_mask| 66845
Token indices sequence length is longer than the specified maximum sequence length for this model (679 > 512). Running this sequence through the model will result in indexing errors
Sequence labeling completed : |labels| 66845
Encoded data saved as data/encoded/test.klue_roberta-base.encoded.pickle 
```

- '--with_text'를 추가하면 원문 문장을 포함하여 저장합니다.

### Train (Fine-Tuning)
인코딩이 완료된 데이터셋을 사용하여 학습을 진행합니다.
``` bash
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --use_kfold --n_splits 5
```

- "--use_kfold"를 사용하는 경우 "--n_splits"와 사용할 폴드 수를 추가합니다.
- "--use_kfold", "--n_splits"를 사용하는 경우, "--fold_i"와 fold 번호를 추가하여 특정 fold에 대해서만 학습을 진행할 수 있습니다.

### Fine-Tuning Details
- Total train data : 303028
- Train : validation = 8 : 2 (242422 : 60606)
- Batch size, epochs : 16, 1
- n-Fold : 5
- total iterations : 폴드별 15152번으로 동일하게 설정


### Inference
트레이닝과 동일하게 전처리한 테스트 데이터셋(66845개)에 대해 모델별, 폴더별 5개 체크포인트 결과의 평균을 최종값으로 하는 앙상블 기법을 적용합니다. --model_folder는 각 모델의 5개 체크포인트 결과가 들어있는 폴더이고, --test_file은 테스트 파일 이름입니다. 

```bash
python inference_ensemble.py --model_folder ./model -- test_file ./test_klue_roberta-base.encoded.pickle > ./results/roberta_output.tsv
```

## Evaluation
- Entity-level micro F1 (Entity F1) 
- 테스트 데이터의 인퍼런스 결과 klue/bert-base가 근소한 성능으로 우수

|PLMs|F1 Score|Accuracy|
|-|-|-|
|klue/bert-base|0.895|0.975|
|klue/roberta-base|0.894|0.974|
|skt/kobert-base-v1|0.861|0.960|
|monologg/koelectra-base-v3-discriminator|0.888|0.972|
|monologg/kobigbird-bert-base|0.889|0.972|


## Changes
- encoding.py에서 sequence labeling을 offset_mappings에 따라 수행하도록 수정했습니다.
- encoding.py 실행 결과물에 모델 정보를 포함하고, hf_trainer.py에서 --pretrained_model_name을 입력하지 않아도 되도록 수정했습니다.
- hf_trainer.py에서 k-fold cross validation을 사용하는 경우 하나의 fold를 지정하여 학습할 수 있도록 수정했습니다.
- 수정된 코드를 활용한 실험의 경우, batch size와 epoch 수를 낮추고 iteration은 동일하도록 맞추었습니다.


## Reference
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- 황석현 외, BERT를 활용한 한국어 개체명 인식기, 한국정보처리학회, 2019
- 유현조 외, 딥러닝 기반 한국어 개체명 인식의 평가와 오류 분석 연구, 한국언어학회, 2021
- Kihyun Kim, Simple Neural Text Classification (NTC), [GitHub](https://github.com/kh-kim/simple-ntc)
- Boseop Kim, NLP Tutorials: Token Classification - BERT, [GitHub](https://github.com/seopbo/nlp_tutorials/blob/main/token_classification_(klue_ner)_BERT.ipynb)
