
# RESAMPLING
- 훈련셋에서 반복적으로 표본을 추출하고, 각 표본에 관심있는 모델을 다시 적합해 적합된 모델에 대해 추가적인 정보를 얻는 것<br/>

---

### 1. Cross-validation
- validation set error rate로 test set error rate를 추정(실제로는 우리게에 test set은 존재하지 않는다.)
- 모델의 성능을 평가하는 model assessment, 모델의 유연성을 선택하는 과정은 model selection
- 회귀분석의 경우 MSE, 분류의 경우 오차율을 기준으로 삼는다.

#### a. Validation set approach
    - 관측치들을 임의로 두 부분, training set과 validation set(hold-out set)으로 나눈다.
    - training set으로 모델 적합 후, validation set에 적용해 MSE를 구한다. 이 때, 두 set의 분할에 따라 MSE의 변동이 매우 크다.

#### b. LOOCV(leave-one-out cross-validation)
    - 관측 데이터 셋을 두 부분으로 분할하되, n-1개 data로 모델 적합 후 제외된 데이터에 적용해 MSE를 구한다.
    - CV = 1/n(sum(all MSE))
    - 편향이 적고, 여러번 수행해도 항상 동일한 결과 얻어진다.
#### c. k-fold CV
    - CV = 1/k(sum(all MSE))
    - 계산량이 적고, 편향-분산 절충 관점에서 LOOCV 대비 적은 분산을 갖는 이점이 있다.(상관성 높은 값들의 평균은 분산이 큼)
    
#### ex) CV로 산출된 test MSE 곡선에서 최소값의 위치(유연성 - 다항식 포함 정도)를 파악해 각 모델의 '성능을 평가'하고 특정 '모델을 선택'한다.

![](https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-1-4614-7138-7_5/MediaObjects/978-1-4614-7138-7_5_Fig4_HTML.gif)

---

### 2. Bootstrap
- 임요한 교수님: 똑같은 모집단을 대상으로 복원 추출을 진행. 이 모집단과 똑같은 성질을 가진 집단 100개를 만들어 통계량을 구한다.<br/>
  ex) 모집단의 var(beta hat): 100개의 부트스트랩된 복제 모집단에서 100개의 beta hat 구해 표본분산을 구한다.(추정량의 분산= quality)
- 추정량 또는 통계학습방법과 연관된 불확실성을 수량화하는데 사용. ex) 선형회귀적합에서 계수의 표준오차 추정
