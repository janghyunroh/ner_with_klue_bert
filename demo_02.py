
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from kobert_tokenizer import KoBERTTokenizer



class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner
        

    def __call__(self, text, list_of_input_ids, list_of_pred_ids):
        # input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        # input_token = self.tokenizer.decode(list_of_input_ids[0])
        # self.encodings = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')

        input_token = self.tokenizer.convert_ids_to_tokens(list_of_input_ids[0])
        input_token = self.modify_tokens(input_token)

        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # print(input_token)
        # print(pred_ner_tag)

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-2:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag})
                    # list_of_ner_word.append({"word": entity_word, "tag": prev_entity_tag})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag})
                    # list_of_ner_word.append({"word":entity_word, "tag":entity_tag})
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
            if i == 0 or i == len(pred_ner_tag)-1: # remove [CLS], [SEP]
                continue
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-2:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence
    
    def modify_tokens(self, tokens):
        # 수정된 토큰들을 저장할 리스트
        modified_tokens = []

        first_token = tokens[0]
        if not first_token.startswith('▁'):
            modified_tokens.append('▁' + first_token)
        else:
            modified_tokens.append(first_token)

        # 토큰 리스트를 두 번째부터 순회하면서 조건에 따라 수정
        for i in range(1, len(tokens)):
            token = tokens[i]
            if token.startswith('##'):
                # '##'으로 시작하는 경우 '##' 제거
                modified_tokens.append(token[2:])
            elif token not in ['[CLS]', '[SEP]']:
                # '[CLS]', '[SEP]'이 아닌 경우 앞에 '_' 추가
                modified_tokens.append('▁' + token)
            else:
                # 그 외의 경우 (즉, '[CLS]', '[SEP]') 원본 유지
                modified_tokens.append(token)

        # 결과 출력
        # print(modified_tokens)
        return modified_tokens



FOLD = 5

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_name_split = model_name.split('/')
    model = []
    model.append(torch.load("./models/klue_roberta-base.15_epochs.100_length.0_fold.pth",map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    # for i in range(FOLD):
    #     if model_name=='monologg/kobigbird-bert-base':
    #         model.append(torch.load(f"/home/user/ner_project/ner/models/{model_name_split[0]}_{model_name_split[1].split('-')[0]}.1_epochs.512_length.{i}_fold.pth",
    #                             map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
    #     else:
            # model.append(torch.load(f"/home/user/ner_project/ner/models/{model_name_split[0]}_{model_name_split[1].split('-')[0]}.2_epochs.512_length.{i}_fold.pth",
            #                     map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))

    return model

@st.cache(allow_output_mutation=True)
def load_token_and_model(select_plm, index_to_label):
    if select_plm=='skt/kobert-base-v1':
        temp_tokenizer = KoBERTTokenizer.from_pretrained(select_plm)
    else:
        temp_tokenizer = AutoTokenizer.from_pretrained(select_plm)
    temp_model = AutoModelForTokenClassification.from_pretrained(select_plm,num_labels=len(index_to_label))
    return temp_tokenizer, temp_model



def inference_for_demo(raw_text, plm_model, bert_best, tokenizer, index_to_label):
    with torch.no_grad():
        encoding = tokenizer(raw_text,return_tensors='pt')
        device = next(plm_model.parameters()).device
        tokenize_text = tokenizer.tokenize(raw_text)
        
        x = encoding['input_ids']
        x = x.to(device)
        mask = encoding['attention_mask']
        mask = mask.to(device)
        predictions = None

        for bert in bert_best:
            # evaluation mode,
            plm_model.eval()
            # Declare model and load pre-trained weights.
            plm_model.load_state_dict(bert,strict=False)

            # Take feed-forward
            y_hat=plm_model(x, attention_mask=mask).logits
            if predictions is None:
                predictions=y_hat
            else:
                predictions+=y_hat
            
    prediction = predictions /5.
    prediction = F.softmax(prediction, dim=-1)
    indice = torch.argmax(prediction,dim=-1)
    result = {}

    list_of_input_ids = []
    list_of_pred_ids = []
    
    # print(indice)
    # for i in range(1, len(indice[0])-1):
    #     result[tokenize_text[i-1]] = index_to_label[int(indice[0][i])]

    for i in range(0, len(indice[0])):
        list_of_pred_ids.append(int(indice[0][i]))
    
    for i in range(0, len(x[0])):
        list_of_input_ids.append(int(x[0][i]))
    
    return list_of_input_ids, list_of_pred_ids


def main():
    # model_name = st.selectbox('Select PLM', ('klue/bert-base', 'klue/roberta-base', 'skt/kobert-base-v1',
    #                           'monologg/koelectra-base-v3-discriminator', 'monologg/kobigbird-bert-base'))

    model_name = 'klue/roberta-base'
    saved_data = load_model(model_name)
    bert_best = [model['bert'] for model in saved_data]
    index_to_label = saved_data[0]['classes']

    tokenizer, plm_model = load_token_and_model(model_name, index_to_label)

    st.title("개체명 인식")

    activitied = ["NER Checker"]

    st.subheader("Input Text to Tokenize")
    raw_text = st.text_area("Enter Text Here", "Type Here")
    if st.button("Enter"):

        # result = inference_for_demo(
        #         raw_text, plm_model, bert_best, tokenizer, index_to_label)
        # st.write(result)

        list_of_input_ids, list_of_pred_ids = inference_for_demo(raw_text, plm_model, bert_best, tokenizer, index_to_label)
        list_of_input_ids = [list_of_input_ids]
        list_of_pred_ids = [list_of_pred_ids]

        decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_label)
        list_of_ner_word, decoding_ner_sentence = decoder_from_res(text = raw_text, list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
        # print("Output:", decoding_ner_sentence)

        st.write(decoding_ner_sentence)
        st.write(list_of_ner_word)

if __name__ == "__main__":
    main()