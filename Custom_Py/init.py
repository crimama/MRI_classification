class init():
  # 이미지 확인용, 매번 치기 귀찮아서 
  def path_img(path):
    import matplotlib.pyplot as plt 
    import cv2 
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.imread(path))
    plt.show()

  #해당 df에 특정 키 값의 이미지 갯수가 몇장인지 확인하는 용 
  def check_images(dir_df,key):
    length = dir_df[dir_df['key']==key]['dir'].values.shape[0]
    return length
#이미지 장수 조절하는 용 
  def droped_indexes(dir_df,length_df,index,standard):
    #해당 인덱스의 키 호출 
    temp_key = length_df.iloc[index]['key']
    #해당 키의 사진 장수 계산 
    temp_length = length_df.iloc[index]['image_length']
    #몇장의 사진을 조절해야 하는지 연산 
    length_diff = abs(int(standard-temp_length)) 
    #dir_df 에서 해당 키의 dir 리스트 불러 옴 
    temp_indexes = dir_df[dir_df['key'] == temp_key].index
    # dir 리스트 중에서 19장만 추림 
    indexes = temp_indexes[:-length_diff]
    return indexes

  #dir_df 만드는 용 
  def dir_df(folder_dir):
    from tqdm import tqdm
    from glob import glob
    import pandas as pd 
    image_name = []
    for i in folder_dir:
      image_name.extend(glob(f'{i}/*.jpg'))
    image_name.sort()

    dir_df = pd.DataFrame(image_name)
    dir_df['Key_id'] =0
    dir_df.columns = ['dir','key']

    #디렉토리에서 마지막 이미지 번호만 때옴 
    for i in range(len(dir_df)):
      dir_df['key'][i] =  dir_df['dir'][i].split('/')[-1]

    #Key 값 + 이미지 순번 되어 있던 거 Key 값만 남김 
    dir_df['key'] = dir_df['key'].apply(lambda x:x[:-8])

    #key값이 8보다 작은 것들 인덱스
    temp = dir_df[dir_df['key'].apply(lambda x:len(x))==8].index

    #8자리인 key값들 0 붙여서 9자리로 만듬 
    dir_df.loc[dir_df['key'].apply(lambda x:len(x))==8,'key'] = dir_df.iloc[temp]['key'].apply(lambda x: '0'+x)
    return dir_df
    
#csv 파일 zscore 임베딩 -> 0과 1로 바꿈 
  def zscore_Embedding(value):
    if value >= -1.0:
      result = 1
    else:
      result = 0
    return result

  target_length = [4,5,6,7,8] #<- 수정해야 하는 병록번호 갯수들 
  def key_mismatch(dir_df,snsb_df,target_length):
    #병록 번호의 length 가 target_length에 속하는 것들 모두 0 더해서 9자리로 맞춤 
    for length in target_length: 
      difference = 9 - length #추가해줘야 하는 자릿수 
      add = '0' *difference #추가해줘야 하는 자릿수 만큼 0추가 
      
      modified_id = snsb_df.loc[snsb_df['병록번호'].map(len)==length,'병록번호'].apply(lambda x : add+x)
      snsb_df.loc[snsb_df['병록번호'].map(len)==length,'병록번호'] = modified_id

    #위 로직으로 보정되지 않은 것들 임베딩으로 보정 아래 dict keys 값들이 보정되지 않은 값임 
    mis_keys_dict = {'17043':'170435316','19021':'190215496','20004':'200045636','97026':'970266683','094018957':'940189576'}
    mis_keys_keys = list(mis_keys_dict.keys())

    #위 mis_keys_dict를 토대로 값 보정 진행 
    temp = dir_df.loc[dir_df['key'].apply(lambda x : x in mis_keys_keys),'key'].apply(lambda x:mis_keys_dict[x])
    dir_df.loc[dir_df['key'].apply(lambda x : x in mis_keys_keys),'key'] = temp
    return dir_df,snsb_df

