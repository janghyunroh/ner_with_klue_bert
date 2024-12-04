# KLUE-BERT 모델을 이용한 법률 문서 개체명 인식기 개발

이 프로젝트는 한국어 Pre-trained Language Model을 활용하여 법적 문서에 포함된 개체명(entity)을 인식하는 시스템을 개발하는 것입니다. 
대상 문서에는 판결문, 청구전조사서 등 다양한 법적 문서가 포함되며, 특정 task를 위해 사전 정의된 레이블을 기반으로 개체명 인식을 수행합니다. 
본 프로젝트는 주로 법적 문서의 자동화된 분석 및 처리에 활용될 수 있습니다.

HugggingFace 라이브러리를 이용해 KLUE-BERT 모델을 Fine-tuning 하였으며, 보안 문제로 직접 데이터 레이블링과 데이터셋 구축을 진행했습니다. 

깃허브에 존재하는 기존 코드를 활용하기 위해 학습 시 여러 번의 전처리를 거치게 됩니다. 

**참고 저장소**
 - [PLM 기반 한국어 개체명 인식 (NER)](https://github.com/ai2-ner-project/pytorch-ko-ner)
 - [PLM 기반 한국어 개체명 인식 (NER) (K-Fold Ver.)](https://github.com/sim-so/pytorch-ko-ner-v2)
## 프로젝트 개요
본 프로젝트는 한국어 자연어 처리(NLP)를 위한 Pre-trained Language Model을 기반으로, 개체명 인식(Named Entity Recognition, NER) 작업을 수행합니다. 이를 위해 문서에서 특정 개체를 추출하고, 해당 개체를 사전 정의된 카테고리에 맞게 분류합니다. 특히, 법적 문서의 특성을 반영한 개체명 인식기를 개발하는 것이 주요 목표입니다.

## 특징
 - Pre-trained Language Model 활용: 한국어에 최적화된 사전 학습된 언어 모델을 사용하여 성능을 극대화합니다.
 - 법적 문서에 특화된 NER: 판결문, 청구전조사서 등 다양한 법적 문서에 적합한 개체명 인식기를 개발합니다.
 - 사전 정의된 레이블: 법적 문서에서 중요한 정보가 포함된 개체명을 정의하여, 이를 인식하고 분류합니다.
## 사용된 기술
 - 한국어 Pre-trained Language Model: BERT, KoBERT, 또는 ELECTRA와 같은 모델을 사용하여 언어 이해 능력을 향상시킵니다. 최종적으로 KLUE-BERT를 차용하게 되었습니다. 
 - PyTorch: 모델 학습 및 추론을 위해 PyTorch를 사용합니다.
 - Hugging Face Transformers: Pre-trained 모델을 로딩하고 Fine-tuning하기 위해 사용됩니다.

## 데이터셋 및 레이블
 보안상의 이유로 데이터셋과 레이블에 대한 구체적인 정보는 공개되지 않습니다. 본 프로젝트에서 사용된 데이터셋은 법적 문서로 구성되며, 각 문서에서 다양한 개체명들을 추출하여 학습에 활용됩니다. 사용자 본인이 적절한 법적 문서 데이터를 준비하여 모델을 훈련시킬 수 있습니다. 각 개체명은 특정한 법적 의미를 지닌 정보로 분류되며, 모델을 사용할 때는 해당 레이블을 기반으로 문서에서 개체를 추출합니다.

## How to Use

**1. 환경 설정**

- python 3.8 기준으로 테스트
- 설치 모듈 상세정보는 requirements.txt 파일 참고 

```bash
pip install -r requirements.txt
```

 - 또는 yaml 파일을 이용해 가상환경 생성 및 activate 
```bash
conda env create -f ner-v2.yaml
conda activate ner-v2
```

**2. 데이터 전처리(Jupyter Notebook 사용)**

 - json_to_tsv_01.ipynb (json → raw)
 - preprocess_00.ipynb (raw → dataset)
 - encoding_03.ipynb (dataset → encoded)

**3. 모델 학습**
```bash
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle
python hf_trainer.py --model_folder models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --use_kfold --n_splits 5
```

**4. Demo**
```bash
streamlit run demo_02.py
```

## Reference
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- 황석현 외, BERT를 활용한 한국어 개체명 인식기, 한국정보처리학회, 2019
- 유현조 외, 딥러닝 기반 한국어 개체명 인식의 평가와 오류 분석 연구, 한국언어학회, 2021
- Kihyun Kim, Simple Neural Text Classification (NTC), [GitHub](https://github.com/kh-kim/simple-ntc)
- Boseop Kim, NLP Tutorials: Token Classification - BERT, [GitHub](https://github.com/seopbo/nlp_tutorials/blob/main/token_classification_(klue_ner)_BERT.ipynb)
