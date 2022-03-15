def make_df():
  import os 
  import pandas as pd 
  import cv2 
  import pandas as pd 
  from glob import glob
  from tqdm import tqdm
  from init import init #초기 설정용 커스텀 패키지 
  import numpy as np 
  import matplotlib.pyplot as plt 
  columns = ['성명',
  '병록번호',
  '만나이',
  '검사일',
  '교육년수',
  'SVLT_recall_total_score_z',
  'SVLT_Delayed_recall_z',
  'SVLT_recognition_score_z',
  'RCFT_immediate_recall_z',
  'RCFT_delayed_recall_z',
  'RCFT_recognition_score_z',
  'K_MMSE_total_score_z',
  'SNSB_II_Domain_Attention_z',
  'SNSB_II_Domain_Language_z',
  'SNSB_II_Domain_Visuospatial_z',
  'SNSB_II_Domain_Memory_z',
  'SNSB_II_Domain_Frontal_z']

  snsb_df = pd.read_csv('/content/drive/MyDrive/DataSet/202202_김예림/SNSB_integerated.csv')
  snsb_df = snsb_df.drop([2098,2591]) #병록번호 없는 행 제거 <- 결측치 
  snsb_df['병록번호'] = snsb_df['병록번호'].apply(lambda x : str(int(x))) #소수점으로 되어 있는 것 처리하고 str형 변형 

  snsb_df = snsb_df[columns]

  ## 결측치 처리 

  #컬럼 중 isnull 검색 
  #검색한 표본들의 평균 나이 
  #평균 나이로 원래  csvs 검색 
  #해당 나이로 검색 된 것들의 평균 값으로 insull 채움 

  null_columns = ['RCFT_immediate_recall_z',
                  'SVLT_recall_total_score_z',
                  'SVLT_Delayed_recall_z',
                  'SVLT_recognition_score_z',
  'RCFT_delayed_recall_z',            
  'RCFT_recognition_score_z',        
  'K_MMSE_total_score_z',           
  'SNSB_II_Domain_Attention_z',       
  'SNSB_II_Domain_Language_z',        
  'SNSB_II_Domain_Visuospatial_z',    
  'SNSB_II_Domain_Memory_z',          
  'SNSB_II_Domain_Frontal_z']         

  for null_column in null_columns:
    try:
      null_age = int(np.mean(snsb_df[snsb_df[null_column].isnull()]['만나이']))
      null_value = np.mean(snsb_df[snsb_df['만나이']==null_age][null_column].dropna())
      snsb_df.loc[snsb_df[null_column].isnull(),null_column] = null_value
    except:
      pass

  #폴더 디렉토리
  # - 환자  별 mri 디렉토리 데이터 프레임 
  # - 이미지 장수 정리 안된 데이터 프레임 
  folder_dir = glob('/content/KYR_B_imaging/*')
  dir_df = init.dir_df(folder_dir)
  dir_df = dir_df.sort_values(by=['key','dir'])
  dir_df = dir_df.reset_index().drop(columns = 'index')

  ### 병록번호- key 값 보정
  # - 병록번호 == key 하게 만들어야 함 -> primary key 
  # - 중복 제거, @@@-1 형태 제거, 9자리로 통일 

  target_length = [4,5,6,7,8] #<- 수정해야 하는 병록번호 갯수들 

  dir_df,snsb_df = init.key_mismatch(dir_df,snsb_df,target_length)
  #인덱스 초기화 
  dir_df = dir_df.reset_index().drop(columns = 'index')
  # plt.hist(snsb_df['병록번호'].map(len))
  # plt.show()

  #에러값들 제외 
  errors = ['000758836-1' '030338456-1' '040145556-1' '050236926-1' '090374796-1'
  '100402746-1' '120010356-1' '870133669-1' '920033543-1' '930236559-1'
  '930257597-1' '970682600-1']

  dir_df = dir_df.drop(np.where(dir_df['key'].map(len)>9)[0])

  # ## new_dir_df 생성 - 이미지 장수 통일 
  # - 위에서 만든 dir_df에서 각 key 별로 이미지 19장으로 맞춤 
  # - 19장으로 맞추는 이유는 가장 비율이 높아서 

  ### key 값 별 이미지 장수 연산 

  #이미지 장수 연산 
  keys = list(set(dir_df['key'])) 
  images_length = pd.DataFrame(keys)[0].apply(lambda x: init.check_images(dir_df,x)) #키 값별로 이미지의 장수들 계산 
  # images_length.value_counts()

  length_df = pd.DataFrame([keys,images_length]).T
  length_df.columns = ['key','image_length']
  length_df['image_length']  = length_df['image_length'].map(int)

  #Standard를 기준으로 less over 분할 
  standard = 19 # 통일 시킬 이미지 장 수 기준 

  # 조정이 필요한 장 수 초과하는 것들만 추림 
  length_df_over = length_df[length_df['image_length']>standard]

  #19장 보다 적은 경우 
  # length_df_less = length_df[length_df['image_length']<standard]

  len(length_df_over)

  ### key값 별로 이미지 19장 있는 new_dir_df 생성 

  new_dir_df = pd.DataFrame(np.zeros(len(length_df_over)*19*2).reshape(-1,2))
  new_dir_df.columns = dir_df.columns

  for index in tqdm(range(len(length_df_over))):
    new_dir_df.iloc[index*19:(index+1)*19,:]  = dir_df.loc[init.droped_indexes(dir_df,length_df_over,index,standard=19)]
  new_dir_df = new_dir_df.sort_values(by=['key','dir'])
  new_dir_df = new_dir_df.reset_index().drop(columns = 'index')

  ## csvs 생성 
  #  - 중복 제거 
  #  - z score -1.0를 기준으로 0(정상) 1(비정상)으로 분류 

  #19장의 사진으로 추린 new_dir_df로 새로운 키 리스트만듬) 

  key_lists = np.unique(new_dir_df['key'])
  key_lists.sort()

  Embedding_columns = snsb_df.columns[5:]

  for column in Embedding_columns:
    snsb_df[f'E_{column}'] = snsb_df[column].map(init.zscore_Embedding)

  #정렬 
  snsb_df = snsb_df.sort_values(by=['병록번호'])
  new_dir_df = new_dir_df.sort_values(by=['key','dir'])

  #snsb_df 중 key_lists에 있는 행들만 추림 -> csvs 
  csvs = snsb_df[snsb_df['병록번호'].apply(lambda x : x in key_lists)].sort_values(by=['병록번호'])
  csvs = csvs.reset_index().drop(columns = 'index')

  #중복 제거 : 이름으로 정렬 후 중복 제거 
  csvs = csvs.sort_values(by=['성명'])
  csvs = csvs.drop_duplicates(['병록번호'], keep = 'first')

  #다시 병록번호 로 정렬 해서 csvs 와 new_dir_df 순을 맞춤 
  csvs = csvs.sort_values(['병록번호'])
  csvs = csvs.reset_index().drop(columns = 'index')
  new_dir_df = new_dir_df.sort_values(by=['key','dir'])
  new_dir_df = new_dir_df.reset_index().drop(columns = 'index')

  #맞춤 확인 
  print(len(csvs),len(np.unique(new_dir_df['key'])))

  return csvs, new_dir_df,key_lists