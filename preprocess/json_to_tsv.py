import os
import pickle
import json
import argparse

from tqdm import tqdm
import pandas as pd


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_path',
        required=True,
        help="Path of file or directory to load original corpus."
    )
    p.add_argument(
        '--save_path',
        required=True,
        help="Path to save data."
    )
    p.add_argument(
        '--return_tsv',
        action='store_true',
        help="If not true, only pickle file will be saved."
    )

    config = p.parse_args()

    return config


def json_to_tsv(file: str):
    cols = ['sentence_id', 'sentence', 'ne'] # 데이터 프레임 생성
    df = pd.DataFrame(columns=cols) # 데이터 프레임 생성
    id = 0

    with open(file) as f:
        DATA = json.loads(f.read()) # 파일을 열어서

    ne = [] # 이중 딕셔너리는 for 문을 돌면서 데이터 프레임에 추가하기 까다롭기 때문에, 전부 리스트에 옮긴 뒤 한 번에 저장한다.
    
    print(DATA)
    
    for document in tqdm(DATA['document']):
    	print(document)
    
    for document in tqdm(DATA['document']):
        for sentence in document['sentence']:
            df.loc[id, 'sentence_id'] = sentence['id'] # sentence_id는 이 문장의 출처와 일련번호를 담고 있다.
            df.loc[id, 'sentence'] = sentence['form'] # 문장 전체
            labels = dict() # 
            for entity in sentence['NE']: # 일련의 엔티티가 
                key = entity['id'] # entity['id']는 1, 2, 3, 4 ... 이렇게 증가
                entity.pop('id') # 이것을 삭제하고
                labels[key] = entity # {1: entity}
            ne.append(labels) # 번호와 엔티티가 짝지어진 딕셔너리를 en 리스트에 하나씩 추가
            id += 1 # id를 하나 증가
    df['ne'] = ne # 각각의 엔티티를 데이터 프레임에 한 번에 추가
    
    return df


def main(config):
    load_path = config.load_path # json 파일이 담긴 폴더 혹은 파일 그 자체
    fn = os.path.split(config.load_path)[1] # config.load_path에서 파일명만 분리하여 fn 변수에 할당합니다. os.path.split() 함수는 경로를 디렉토리 부분과 파일명 부분으로 나눕니다. 이 중 파일명 부분만을 fn에 저장합니다.
    if fn.rfind(".") > 0: # 파일명 fn에서 확장자를 제거합니다. rfind(".")는 파일명에서 마지막으로 나타나는 .의 위치를 찾습니다. 만약 .이 발견되면, 파일명에서 그 위치까지를 잘라서 확장자 없는 파일명을 fn으로 저장합니다.
        fn = fn[:fn.rfind(".")]
    save_fn = os.path.join(config.save_path, fn) # 변환된 데이터의 결과물을 저장할 경로를 save_fn 변수에 할당합니다. config.save_path에 지정된 디렉토리 경로와 앞에서 만든 파일명 fn을 합쳐서 저장합니다.

    if os.path.isdir(load_path): # load_path가 디렉토리인지 확인합니다. 만약 디렉토리라면 그 안에 있는 모든 파일을 반복적으로 처리합니다.
    # os.listdir(load_path)는 load_path 디렉토리 안의 모든 파일 목록을 반환합니다.
        dfs = []
        for file in tqdm(os.listdir(load_path)): # tqdm을 사용해 파일 목록을 처리하면서 진행 상태를 시각적으로 표시합니다.
            df = json_to_tsv(os.path.join(load_path, file))
            dfs.append(df)
            # 각 파일을 json_to_tsv() 함수로 읽어들여, 그 결과를 데이터프레임 df로 변환한 뒤 dfs 리스트에 추가합니다.
        data = pd.concat(dfs) # 모든 파일의 데이터프레임을 pd.concat(dfs)로 하나의 데이터프레임 data로 결합합니다.
    else:
        data = json_to_tsv(load_path) # 만약 load_path가 파일이라면, 단일 파일을 json_to_tsv() 함수로 처리해 data에 할당합니다.

    with open(save_fn+'.pickle', "wb") as f: # 변환된 데이터를 피클(pickle) 파일로 저장합니다. 파일명은 save_fn에 .pickle 확장자를 붙인 것입니다. 
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump() 함수는 데이터를 피클 형식으로 직렬화하고, protocol=pickle.HIGHEST_PROTOCOL을 사용해 가능한 최고의 프로토콜을 사용하여 저장합니다.

    if config.return_tsv: # config.return_tsv가 True인 경우, 데이터프레임 data를 TSV 파일로 저장합니다.
        data.to_csv(save_fn+'.tsv', sep='\t', index=False)
        # 파일명은 save_fn에 .tsv 확장자를 붙인 것이며, 탭(\t)으로 구분된 형식으로 저장됩니다.
        # index=False로 설정하여, 데이터프레임의 인덱스는 파일에 포함되지 않도록 합니다.

if __name__ == '__main__':
    config = define_argparser()
    main(config)
