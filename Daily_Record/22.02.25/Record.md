1. 모델 축소 + RES50 + Dense169 + Effb4 + albumentation 
    - 시간 : 390s/e
    - train : 0.65
    - vald : 0.5
2. 모델축소 + effv4 + resv2 151 + vgg제거 + albumentation 
    - 시간 : 420s/e
    - train : 0.6694
    - test : 0.5125
3. 모델축소 + effv4 + resv2 152 + dense 169 + lr : 0.0001 + albumentation 
    1. 시간 : 409 
    2. train : 0.6187
    3. test : 0.5708 
4. 모델축소 + effv4 + resv2 151 + vgg제거 + lr : 0.0001 + albumentation 
    1. 시간 : 406
    2. train : 0.6545
    3. valid : 0.5833
5. 모델축소 + effv4 + resv2 151 + vgg제거 + albumentation 
    1. 시간 : 382
    2. train : 0.58
    3. valid : 0.55
6. res50 + Dense2121 + eff4 + albumentation 
    1. 시간 : 444
    2. train : 0.7
    3. valid : 0.4833
