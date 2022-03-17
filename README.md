## 프로젝트 주요 특징 
- 3 모델 앙상블
- 19 채널 
- Augmentation 
![image](https://user-images.githubusercontent.com/92499881/158784451-1163393b-e98a-487a-b74a-551cb2abdc0f.png)

## 
- **프로젝트 목적**
    - 환자의 Brain MRI 이미지와 나이, 교육년수 데이터들을 이용해 환자의 건강 상태 통계 score를 예측하는 모델 개발
- **프로젝트 설명**
    - Input 데이터 : 뇌 MRI 이미지, 만 나이, 교육 년수
    - output 데이터 : Recall_total_score_z, Delayed_recall_z, Recognition_score_z 등 총 12개의 z 스코어
    - z 스코어가 -1보다 작은 경우 하위 15%를 의미하므로 비정상으로 판단 하여 해당 기준으로 환자의 건강 상태 정상과 비정상을 분류

## 2. 전처리

### 1. 병록 번호

- 환자 별로 고유의 병록 번호(key)값이 있음
- 제너레이터에서 데이터 호출 시 병록번호를 이용해 이미지와 속성 데이터를 호출 함 ( 병록번호(key)가 Primary key 역활을 함 )
- **key 값 보정**
    - 각 Brain MRI 이미지에 환자의 병록번호가 이미지 이름으로 되어 있음
    - 대부분의 병록번호(key)값은 9자리이지만 몇몇 환자의 경우 병록 번호가 4~8자리인 경우가 있음 —→ 속성 데이터와 연결을 위해 9자리로 통일해야 함
    - 앞에 0을 추가하여 9자리로 통일 함
- **결측치 제거**
    - 병록번호가 9자리가 아닌 @@@-1 형태로 되어 있는 경우가 있음 ——> 이 경우 중복 데이터로 모두 제거 함
- **Key list 생성**
    - 속성 테이블과 이미지를 연결할 key_lists를 따로 생성 함
    - 제너레이터에서 이 key_lists를 이용해 train-valid-test를 분리하고, 이미지와 속성 데이터를 호출 함

### 2. 이미지 데이터

- **이미지 장수 통일**
    - 환자 별로 저장되어 있는 Brain MRI 이미지의 장수가 모두 다름
    - 20장인 경우가 가장 많았지만 최대한 데이터를 보존하기 위해 19장으로 장수를 통일 함
        - 사진이 많은 경우는 단순히 사진을 줄이면 되지만 사진 장수가 적은 경우는 버려야 하기 때문에 19장을 선택
        - 19장보다 많은 경우 사진을 선택하는 기준은 사진 이름으로 정렬한 뒤 0 ~18번을 선택
- **컬러 스케일 → Gray Scale**
    - 환자 별 19장의 Brain MRI 이미지를 한번에 처리하기 위해 각각의 19장 이미지를 한개의 이미지로 통합해야 함
    - 19장 이미지를 한개의 이미지로 통합하기 위해 각각의 이미지를 모두 Gray Scale로 변환 함
- **이미지 리사이즈**
    - 환자 별로 이미지의 크기가 모두 다르지만 대체로 shape : (524,524,3) 정도의 크기를 가짐
    - 이미지 사이즈가 너무 커 224 사이즈로 rescale 함
- **이미지 통합**
    - Gray Scale로 변환 된 19장의 이미지를 한개의 이미지로 통합 함
    - 넘파이 stack을 이용해서 shape : (224,224,19) 형태로 이미지를 통합 함
- **Augmentation**
    - Augmentation은 Albumentation을 이용해 Random Augmentation을 적용 함
    - 적용한 Augmentation (아래 요소들이 랜덤하게 적용 됨)
        - JpegCompression(quality_lower=85,
        - quality_upper=100, p=0.5),
        - HueSaturationValue(hue_shift_limit=20,
        - sat_shift_limit=30,
        - val_shift_limit=20, p=0.5),
        - A.HorizontalFlip(p=0.5),
        - Rotate(p=0.5, limit = 90),
        - Crop()
    - 이미지를 Grayscale로 변형하고 19 channel로 합치기 때문에 색상 보다는 이미지 형태 를 변형시키는 방식 위주로 Augmentation 진행 함

**이미지 장수 통계** 

![image](https://user-images.githubusercontent.com/92499881/158146941-8a25f1ae-0199-4e91-bb8e-8c5e5f5f3d53.png)

**이미지 장수 통계** 

![image](https://user-images.githubusercontent.com/92499881/158146913-eaa2e3f4-62f2-4e72-88f7-39cd45bbf4a6.png)

### 3. **속성 데이터**

- **결측치 처리**
    - 예측해야 하는 z score에 결측치가 존재 함
    - 결측치를 보간하기 위해 평균나이를 이용 함
        - 특정 컬럼에서 결측치를 갖는 표본들의 평균 나이 계산
        - 해당 나이를 갖는 전체 표본에서 해당 컬럼의 평균 값을 계산
        - 해당 평균 값을 이용해서 결측치 보간
- **임베딩**
    - 예측해야 하는 z score 컬럼들을 -1을 기준으로 0(정상)과 1(비정상)로 분류 함
    - 0과 1로 분류한 컬럼을 새로 생성 함
- **Min Max Scaling**
    - input 데이터의 경우 “만 나이”, “교육 년도” 에 Scaling이 적용 됨
    - output 데이터의 경우 해당 컬럼에 Min Max Scaling이 적용 됨
    - Scaling에는 Train데이터만 사용 됨

**컬럼 별 결측치 수** 

![image](https://user-images.githubusercontent.com/92499881/158146873-67d458a6-ef28-455f-ab59-f9c903e2a54c.png)

## 3. 모델 설명

### 1. 네트워크 구성

- 모델 네트워크는 크게 4단계로 구성 됨
- 256,256,19 형태의 이미지를 Pretrained model에 적용할 수 있도록 사이즈 변환하는 **Input to pre 구간**
- 변환된 이미지를 3개의 Pretrained model에 통과 시킨 뒤 Concatenate 시켜 앙상블 하는 **Pretrained Model Ensemble 구간**
- 앙상블 된 데이터와 CSV 데이터를 Concatenate 시키는 **CSV Concatenate 구간**
- Concatenate 된 데이터를 최종 데이터로 Regression 하는 **Dense 구간**

### 2. Optimizer

- Optimizer로 Tensorflow Addons API의 **Rectifier Adam**를 사용 함
    - Total steps : 10000
    - warmup_proportion=0.1
    - min_lr=0.00001
    - Adam, SGD, AdamW를 모두 실험을 해 보았지만 RAdam이 가장 좋은 성능을 보여 줌
        - *Classification_score_report.csv 참고*
- Learning rate 조절을 위한 Scheduler로 **Exponential Decay**를 사용
    - Initial Learning rate : 0.001
    - Decay steps : 100000
    - Decay rate : 0.96
    - Stair case : True

### Cross Validation

- Cross Validation 전략으로 Stratified K-fold(K=5) 를 시행 함
- 총 환자의 수는 1606명으로 Test 셋은 약 10%비율로 우선 분할 하고 남은 약 1500명을 K fold 로 Train-Valid를 분할 함

**모델 구성도** 

![image](https://user-images.githubusercontent.com/92499881/158146627-597d66cd-4349-41bf-95de-d88dd24b4d27.png)

### Regression

**기존 방식** 

- 초기에는 softmax 와 sparse categorical cross entropy loss 함수를 이용한 Binary Classification 모델로 진행 함
- 하지만 Train, Valid, Test 모두 Accuracy 가 높게 나오지 않았음 ——> 대게 0.5 ~ 0.6 수준
    - 0과 1을 분류하는 Binary Classification에서 0.5 ~ 0.6의 Accuracy는 학습 자체가 안되었다고 판단 함
    - 학습이 안되는 문제로 임의로 설정한 Threshold 때문이라 생각 함
        - 0(정상) 과 1(비정상)을 나누는 기준으로 통계학적 근거 z score = -1 : 하위 15% 로 설정했기 때문에 실제 환자의 임상학적 정상 비정상 기준 간의 차이가 있을 수 있다고 생각 함
        - 그리고 모델이 학습하기 위해선 Class 0의 일반 적인 패턴, Class 1의 일반적인 패턴을 학습해야 하지만 class 내부의 패턴이 너무 다양해 일반적인 패턴을 학습하지 못한다고 추측 함
- **그래서 softmax로 class 0 과 class 1의 일반적인 패턴을 학습 해 분류하는 방식 대신 값을 직접 예측하고 예측된 값에 Threshold를 이용해 분류하는 방식으로 변경 함**

**Regression** 

- 모델은 동일하게 가져가되 softmax 대신 sigmoid, loss 함수로 mse를 사용 하여 원래 컬럼의 value를 예측하는 방식으로 진행 함
- 그리고 예측된 값을 Threshold 값을 기준으로 크면 0(정상), 작으면 1(비정상)으로 분류 함

**Regression 예측 결과 예시** 

- 오른쪽 분포 그래프는 Train 데이터로 Predict한 결과로 빨강은 이상 데이터, 파랑은 정상 데이터를 의미 함
- 이 그래프의 경우 -1을 Threshold로 잡을 경우 정상과 비정상 간의 분류가 불가능 함
- 하지만 0.3을 Threshold로 하여 분류를 할 경우 정상과 비정상을 분류할 수 있음
- **SVLT_recall_total_score_z 컬럼의 분포**
![image](https://user-images.githubusercontent.com/92499881/158146733-5bee5d40-d99d-49c6-a928-9851fce3befb.png)


                                                 Redline = -1 

- **Regression 예측 결과 예시**
    
    ![image](https://user-images.githubusercontent.com/92499881/158146766-8b966b93-f2da-4eec-ac32-6de66b2e75bb.png)
    

## 결과

- 12개의 각 model.h5 파일
- 각 train, test 별 loss, acc, f1-score
- output 컬럼 별 loss, acc 그래프(train, test)
- output 컬럼 total loss, acc 그래프
- ROC, AUC,AUROC → output 컬럼별 그리고 total 1개

각 모델에서 추출할 것 → 소프트 보팅 한 이후 걸로 진행 

- loss, val_loss, regression_result, roc,auc,auroc
- 각 모델 추출한 것들 모아서 최종 결과 레포트 진행

### 필요한 것들

**각 모델 별** 

1. Train, valid loss 그래프
2. Train, Valid, Test Regression Result 분포 그래프 
3. Train, Valid, Test Score (ACC, Recall, Precision, F1-score) 
4. ROC, AUC, AUROC 그래프 

**통합** 

1. loss 그래프 - Train, Valid 
2. Score 비교 (Acc, Recall, Precision, F1 score) 
3. ROC, AUC, AUROC 비교
