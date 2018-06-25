
## 0. 공통 수행
- summary, plot...
- Continuous variable: hist, boxplot, density, mean, t.test...
- Categorical variable: table, **barplot(table(x)**, binom.test...
- Continuous x and y: plot, cor, lm, lqs...
- Categorical x and Continuous y: ANOVA, boxplot, lm...
- Continuous x and Categorical y: glm(family = 'binom')

-----

## 1. Continuous variables EDA
- 데이터의 정규성 검사: qqplot, qqline() function은 분포가 정규분포와 얼마나 유사한지 검사하는데 사용된다
- 가설검정과 신뢰구간: t.test()로 일변량 t-검정과 신뢰구간 구할 수 있다. 실제로는 데이터의 분포가 정규분포가 아니라도 큰 문제 안된다.
- 이상점 찾기: 로버스트 통계량


```R
library(ggplot2)
library(dplyr)
class(mpg$hwy)
```

    Warning message:
    “package ‘dplyr’ was built under R version 3.4.2”
    Attaching package: ‘dplyr’
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    



'integer'



```R
opar <- par(mfrow = c(2, 2))
hist(mpg$hwy)
boxplot(mpg$hwy)
qqnorm(mpg$hwy)
qqline(mpg$hwy)
par(opar)
```


![png](output_4_0.png)


### a. One sample t.test
- H0: mu <= 22.9 vs H1: mu > 22.9


```R
mu0 <- 22.9
t.test(mpg$hwy, mu = mu0, alternative = 'greater')
```


    
    	One Sample t-test
    
    data:  mpg$hwy
    t = 1.3877, df = 233, p-value = 0.08328
    alternative hypothesis: true mean is greater than 22.9
    95 percent confidence interval:
     22.79733      Inf
    sample estimates:
    mean of x 
     23.44017 



_만약 실제 모 평균 고속도로 연비가 22.9라면(=귀무가설 하에서) 우리가 관측한 것 만큼 큰 표본평균값과 t 통계량이 관측될 확률이 8.3_

    a. t.test 세부 설명 참고
        - <http://www.dodomira.com/2016/04/02/r%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-t-test/>
        - <http://rfriend.tistory.com/127>

### b. Outliers and Robust statistical methods
- 로버스트 통계 방법은 이상점의 영향을 적게 받는 절차
- mean 대신 median, SD 대신 Median Absolute Deviance를 사용


```R
c(mean(mpg$hwy), sd(mpg$hwy))
```


<ol class=list-inline>
	<li>23.4401709401709</li>
	<li>5.95464344116645</li>
</ol>




```R
c(median(mpg$hwy), mad(mpg$hwy))
```


<ol class=list-inline>
	<li>24</li>
	<li>7.413</li>
</ol>



-----

## 2. 성공 - 실패 범주형 변수 분석
- 요약 통계량 계산: table(도수 분포), prop.table(상대도수)
- 데이터 분포의 시각화: barplot()
- 가설검정과 신뢰구간: binom.test() 함수를 이용해 성공률(1, 성공 or 해당)에 대한 검정과 신뢰구간을 구할 수 있다.


```R
set.seed(1606)
n <- 100
p <- .5
x <- rbinom(n, 1, p)

x <- factor(x, levels = c(0,1), labels = c('no', 'yes'))
head(x)
table(x)
opar_2 <- par(mfrow = c(1, 2))
barplot(table(sleep$group))
par(opar)
```


<ol class=list-inline>
	<li>yes</li>
	<li>yes</li>
	<li>yes</li>
	<li>yes</li>
	<li>no</li>
	<li>no</li>
</ol>




    x
     no yes 
     46  54 



![png](output_14_2.png)


_- 우리는 p값 설정을 통해 실제 지지율이 50%임을 알고 있지만, 모른다고 가정 후 검정 진행_<br/>
_- H0: p = 0.5 vs H1: p != 0.5_


```R
binom.test(x = length(x[x == 'yes']), n = length(x), p = 0.5, alternative = 'two.sided')
```


    
    	Exact binomial test
    
    data:  length(x[x == "yes"]) and length(x)
    number of successes = 54, number of trials = 100, p-value = 0.4841
    alternative hypothesis: true probability of success is not equal to 0.5
    95 percent confidence interval:
     0.4374116 0.6401566
    sample estimates:
    probability of success 
                      0.54 



1. 귀무가설 H0: p = 0.5가 참일 때, 주어진 데이터만큼 극단적인 데이터를 관측할 확률은 48%. 귀무가설을 기각할 증거 희박.
2. 95 percent confidence interval의 의미:
    - '모수가 주어진 신뢰구간에 포함될 확률'은 말도 안된다. **모수는 우리가 값을 모를 뿐 상수이다.** 때문에 '모수가 xxx할 확률'은 안된다.<br/>
      이미 구해진 신뢰구간은 참 모수를 포함하든가 아니면 포함하지 않든가 둘 중 하나다.
    - 정확한 정의는 '같은 모형에서 반복해서 표본을 얻고, 신뢰구간을 얻을 때 신뢰구간이 참 모수값을 포함할 확률이 95%가 되도록 만들어진 구간'

----

## 3. 설명변수(explanatory variable)와 반응변수(response variable)
- 보통 인과관계에서 원인이 되는 것으로 믿이지는 변수를 X, 결과가 되는 변수를 Y (_아버지의 키 X - 아들의 키 Y, 약의 종류 및 복용량 X - 효과 Y_)
- explanatory = 예측(predictor) = 독립(independent)
- response = 종속(dependent)

-----

## 4. Continuous X and Y
1. plotting 통해 관계의 모양을 파악. 관계가 선형인지, 세기가 어떤지, 이상치는 어떤지 파악한다.
2. 상관계수를 계산한다. (**상관계수는 '선형'관계의 강도만을 재는 것**, cor() 함수는 기본적으로 Pearson 상관계수 계산(로버스트한 method = 'kendall' 옵션 선택 가능) <br/>
 산점도를 그리지 않고 상관계수만 보는 것은 위험. 다양한 비선형적 관계와 데이터 군집성 등의 패턴 알기 위해 시각화 필요!
3. **선형 모형을 적합한다(규봉: 후에 진단).** 모형의 적합도와 모수의 의미를 살펴본다.
4. 잔차의 분포를 살펴본다. 잔차의 이상점은 없는지, 잔차가 예측(독립, 설명)변수에 따라서 다른 분산을 갖지는 않는지 본다.
5. 이상치가 많을 경우 로버스트 회귀분석을 적용한다.
6. 비선형 데이터에는 LOESS 등의 평활법을 사용한다.


```R
ggplot(mpg, aes(cty, hwy)) + geom_jitter() + geom_smooth(method = 'lm')
par(opar_2)
```




![png](output_22_1.png)


---

### a. Linear regression model fitting

- Yi ~ beta0 + beta1x1 + ... + betanxn ei, ei ~ iidN(0, sigma^2)
- xij는 j번째 설명변수의 i번째 관측치
- iid는 독립이고 동일한 분포를 따름을 나타낸다.
- 모수는 절편(intercept) beta0, 계수(slope) betan, 오차항(error term)의 분포의 분산 sigma^2
- lm() 함수는 위의 선형 모형을 Least Square Method로 추정. 즉, 잔차의 제곱합을 최조화하는 문제를 풀어 추정치를 구한다. [(sum(yi - (b0 +b1xi))^2]
    + 모형적합(model fitting)이란, 관측된 데이터를 사용해 우리가 모르는 beta값을 알아내는 작업
- lm.summary()는 각 추정치와 더불어, 각 모수값이 0인지 아닌지에 대한 가설검정 결과를 보여준다. 즉, 절편에 대한 H0: beta0 = 0 vs H1: beta0 != 0,<br\>계수에 대한 H0: beta1 = 0 vs beta1 != 0에 대한 검정 결과를 보여준다.
- 이 가설들은 t.test로 주어진다. 따라서 추정값(estimate), 표준편차(standard **error**), 그리고 그 비율로서의 t-value(= estimate / SE),<br\> 그리고 적절한 자유도에 대한 P-값(Pr(>|t|))을 보여준다.


```R
summary(lm(mpg$hwy ~ mpg$cty, data = mpg))
```


    
    Call:
    lm(formula = mpg$hwy ~ mpg$cty, data = mpg)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -5.3408 -1.2790  0.0214  1.0338  4.0461 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  0.89204    0.46895   1.902   0.0584 .  
    mpg$cty      1.33746    0.02697  49.585   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 1.752 on 232 degrees of freedom
    Multiple R-squared:  0.9138,	Adjusted R-squared:  0.9134 
    F-statistic:  2459 on 1 and 232 DF,  p-value: < 2.2e-16



_a. hwy 값에 대한 cty의 선형 효과는 통계적으로 유의하다. 귀무가설 하 처럼 beta1 = 0일 경우 지금처럼 t값 관측은 거의 불가.
따라서 귀무가설을 기각_<br/>
_b. F-statistic은 H0: 평균(절편) 외에 '다른 모수는 효과가 없다 vs H1: not H0'이라는 가설에 대한 검정통계량이다.<br/>
    설명변수가 하나인 경우엔 지금처럼 계수에 대한 t-검정 결과와 동일. 하지만 여러개일 때라면 **모든 설명변수를 아울러서 유의성을 검정하는 통계량**_

### b. Model Diagnostics for Regression
- 선형회귀 결과가 의미가 있으려면 다음의 가정 충족해야 한다.<br/>
    a. x와 y의 관계가 선형이다. <br/>
    b. 잔차의 분포가 독립이다. <br/>
    c. 잔차의 분포가 동일하다. <br/>
    d. 잔차의 분포가 N(0, sigma^2)
<br/>
- 일변량 수치변수에서의 t-검정과 유사하게, 가장 만족하기 어려운 조건 d는 못 지킬 수 있다.
- 하지만 조건 c가 어긋나는 경우, 분산이 x 값에 따라 변하는 것은 추정치와 오차의 유효성에 영향을 준다.<br/>
 (이러한 오차분포를 이분산성 오차 분포 heteroscedastic error distribution)<br/>
 이런 경우 보통 가중회귀분석(weighted regression)을 이용한다.
- 조건 a, b가 어긋나면 모든 모수의 의미가 왜곡되게 되므로 시각적으로 잔차의 분포가 x에 따라, y에 따라 변하는지 봐야한다.(regression diagnostic)


```R
par(mfrow = c(2, 2))
plot(lm(hwy ~cty, data = mpg))
par(mfrow = c(1, 1))
```


![png](output_29_0.png)


1. Residual vs Fitted: 잔차의 분포가 예측값과 독립인지 확인
2. Normal Q-Q: 표준화된 잔차의 분포가 정규분포에 가까운지 확인
3. Scale-Location: 잔차의 절대값의 제곱근과 예측값 사이의 관계 확인
4. Residual vs Leverage: 표준화된 잔차와 레버리지 간의 관계 확인<br/>
<br/>

| 회귀 진단 중 중요한 수치중 하나인 레버리지: Leverage hii는 Projection Matrix의 i번째 diagonam matrix.<br/>
        만약 hii 값이 크면 yi 값의 작은 변화에도 적합값 yi hat이 yi 방향으로 더 많이 **끌려가게**된다. 즉, 더 많은 influence 받는다.

### c. 로버스트 선형회귀분석
- 수량형 변수에 이상치가 있을 경우 로버스트 통계방법을 이용하는 것이 좋다.
- **선형회귀분석의 모수 추정값은 '잔차의 분포에서 이상치가 있을 때' 지나치게 민감하게 반응한다. 이는 모수를 추정할 때 최소제곱법을 사용하기 때문이다.**<br/>
  이상치에 민감하지 않은 추정법이 필요할 때에는 로버스트 회귀분석을 사용한다.


```R
library(MASS)
set.seed(123)
lqs(stack.loss ~., data = stackloss)
#lqs() 함수는 데이터 중 '좋은' 관측치만 적합에 사용한다.(제곱오차의 quantile 최소화)
```

    
    Attaching package: ‘MASS’
    
    The following object is masked from ‘package:dplyr’:
    
        select
    



    Call:
    lqs.formula(formula = stack.loss ~ ., data = stackloss)
    
    Coefficients:
    (Intercept)     Air.Flow   Water.Temp   Acid.Conc.  
     -3.631e+01    7.292e-01    4.167e-01   -8.131e-17  
    
    Scale estimates 0.9149 1.0148 




```R
lm(stack.loss ~., data = stackloss)
```


    
    Call:
    lm(formula = stack.loss ~ ., data = stackloss)
    
    Coefficients:
    (Intercept)     Air.Flow   Water.Temp   Acid.Conc.  
       -39.9197       0.7156       1.2953      -0.1521  



* Acid.Conc 변수의 효과에 차이가 있다.

### d. 비선형/비모수적 방법, 평활법과 LOESS
- 선형회귀분석은 모형이 '선형'임을 가정한다.
- 비선형적인 x-y 관계를 추정해내기 위해서는 nonlinear regression이나 polynomial regression을 사용하기도 한다. <br/>
  (규봉: 하지만 이들도 linearizable regression model로 x,y 변수들에 변환을 취하면 linear regression이라 볼 수 있다.)
- 하지만 이러한 모형들보다 손쉽게 사용할 수 있는 것은 모형에 아무 가정도 하지 않는 smoothing이다.
    + yi = f(xi) + ei, ei ~ iid(0, sigma^2)
    + f(x)는 보통 두 번 미분 가능한 것으로 정의하며 선형함수일 필요가 없다.
    + 잔차가 정규분포일 필요가 없다.
<br/>
- 다양한 smoothing 기법 중 local regression 방법인 LOESS(locally weighted scatterplot smoothing)이 선호된다. <br/>
    각 예측변수 x0 값에서 가장 가까운 k개의 (xi, yi) 관측치들을 사용해 2차 다항회귀(ploynomial) 모형을 적합해 f hat(x0)을 추정하고,<br/>
    이것을 다양한 x0 값에 대해 반복하는 것이다. 크기가 변하는 window를 좌에서 우로 이동하며 로컬하게 간단한 모형을 적합하는 것.<br/>
    (평활의 정도인 파라미터 k값은 CV로 최적화)


```R
plot(hwy ~ displ, data = mpg)
mpg_lo <- loess(hwy ~ displ, data = mpg) #loess()
summary(mpg_lo)
```


    Call:
    loess(formula = hwy ~ displ, data = mpg)
    
    Number of Observations: 234 
    Equivalent Number of Parameters: 4.57 
    Residual Standard Error: 3.372 
    Trace of smoother matrix: 4.98  (exact)
    
    Control settings:
      span     :  0.75 
      degree   :  2 
      family   :  gaussian
      surface  :  interpolate	  cell = 0.2
      normalize:  TRUE
     parametric:  FALSE
    drop.square:  FALSE 



![png](output_36_1.png)



```R
ggplot(mpg, aes(displ, hwy)) +
geom_point() +
geom_smooth()
```

    `geom_smooth()` using method = 'loess'





![png](output_37_2.png)


---

## 5. Categorical x, Continuous y
- side-by-side boxplot(병렬상자그림)을 이용해 데이터를 시각화한다. 집단 간 평균, 분산 등에 차이가 있는지 확인
- lm() 함수로 ANOVA 선형 모형을 적합.
- plot.lm()으로 잔차의 분포 확인. 이상점과 모형의 가정 만족하는지 체크

### a. ANOVA(analysis of variance)
- 설명변수가 범주형이고, 반응변수가 수량형일 경우 선형 모형의 특별한 예인 분산분석을 이용(집단의 개수가 2개일 경우 사용하는 two-sample t-test는 특이한 경우)
- 범주 변수 x의 값에 따라 데이터 그룹 i = 1, ..., p로 나누고, 각 그룹에 j= = 1, ..., ni개의 관측치가 있을 때 분산분석 모형은<br/>
    Yij = beta i + eij, eij ~ iid N(0, sigma^2)
- [X design matrix](https://en.wikipedia.org/wiki/Design_matrix)
- 회귀분석, ANOVA 분산분석 모두 수학적으로 동일한 선형 모형. 모수에 대한 검정도 t-test, 적합도 평가 기준도 동일


```R
mpg %>% ggplot(aes(class, hwy)) + geom_boxplot()
```




![png](output_41_1.png)



```R
summary(lm(hwy ~ class, data = mpg))
```


    
    Call:
    lm(formula = hwy ~ class, data = mpg)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -8.1429 -1.8788 -0.2927  1.1803 15.8571 
    
    Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
    (Intercept)       24.800      1.507  16.454  < 2e-16 ***
    classcompact       3.498      1.585   2.206   0.0284 *  
    classmidsize       2.493      1.596   1.561   0.1198    
    classminivan      -2.436      1.818  -1.340   0.1815    
    classpickup       -7.921      1.617  -4.898 1.84e-06 ***
    classsubcompact    3.343      1.611   2.075   0.0391 *  
    classsuv          -6.671      1.567  -4.258 3.03e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 3.37 on 227 degrees of freedom
    Multiple R-squared:  0.6879,	Adjusted R-squared:  0.6797 
    F-statistic: 83.39 on 6 and 227 DF,  p-value: < 2.2e-16



**base model: 2seater**

### b. 분산분석 진단
1. 잔차의 분포가 독립
2. 잔차의 분산이 동일
3. 잔차의 분포가 N(0, sigma^2)


```R
par(mfrow = c(2, 2))
plot(lm(hwy ~ class, data = mpg))
par(mfrow = c(1, 1))
```


![png](output_45_0.png)


---

## 6. Continuous x, Categorical y (success-fail)
1. X와 jitter된 Y 변수의 산점도를 그려본다. 그리고 Y 변수의 그룹별 X 변수의 병렬상자그림을 그려본다.<br/>
    Y 변수 값에 따라 X 변수의 분포가 차이 있는지, 두 변수 간 어떤 관계가 있는지, log-odds(log(mu / (1 - mu)), mu: 성공확률)(=logit)가 어떤지 체크<br/>
        규봉: 왜 선형회귀를 사용하지 않는가?
        a. 확률적으로 볼 때 [0,1] 초과의 문제. [0,1] 안에서 확률값이 산출되도록 p(x) = exp() / 1 + exp()
        b. 이를 약간 변형하면 p(X) / 1 - p(X) = exp(b0 + b1x)이 된다(odds).
        c. 여기에 log 씌우면 로그우도, 로짓 즉 logit(mu) = log(p(X) / 1 - p(X)) = eta(x) = beta0 + beta1 * X.
            (확률값 mu(x)는 선형예측(linear predictor)함수인 xbeta와 logit 함수로 연결되어 있다.)
        d. 이 로짓 변환된 값을 설명변수의 선형함수로 모형화하는 것이 이항분포에 적용된 GLM 모형의 골자.
            GLM에서 로짓변환 함수의 역할을 하는 함수가 링크 함수.
        e. 여기서 회귀 계수의 추정(모수벡터 beta의 추정값 beta hat)은 최대가능도 방법를 활용. 가능도 함수를 최대화하도록 선택(MLE)
            Maximum Likelihood Estimation: l(b0, b1) = ㅠp(xi) ㅠ(1 - p(xi)) 
        ex) X의 한 유닛 증가로 default의 로그 공산은 0.0055 유닛만큼 증가 / 온도가 1도 상승할 때, 로그 오즈비가 0.0055만큼 증가
            추정값 beta hat이 얻어지면 반응변수의 기대값은 선형추정 값인 eta hat = eta hat(x) = x*beta hat은 무한대의 범위 갖는다.
        f. 이를 확률값으로 변환하기 위해서는 '로짓 링크함수의 역함수'인 'logistic fn'을 사용해 다시 [0, 1] 사이의 확률값으로 되돌린다. 
            mu hat(x) = logit^-1(eta hat) = logistic(eta hat) = 1 / 1+exp(-eta hat) = exp(eta hat) / exp(eta hat)+1
        
2. glm() 함수로 일반화 선형 모형 적합
3. plot.glm()으로 잔차의 분포를 살펴본다. 이상점은 없는지, 모형의 가정은 만족하는지 체크

glm() 함수 안에서 family = 'binomial'이 주어졌을 때, 반응변수는 몇 가지 방법으로 표현될 수 있다.
    1. 반응변수 yi가 0에서 1 사이의 숫자이면 yi = si / ai로 간주된다. si = '성공 횟수', ai = '시도 횟수'.
       ai는 옵션에 정의해야 한다. 모든 si = 0 혹은 1이면 weights를 지정하지 않아도 된다.(ai = 1로 간주)
    2. 반응변수가 0-1 숫자 벡터일 경우에는 #1처럼 간주된다(weights = 1). TRUE / FALSE 논리 벡터이면 0 = F, 1 = T로 간주.
       2-레벨 이상의 factor 변수는 첫 번째 레벨이 '실패'로, 나머지 레벨은 '성공'으로 간주
    3. 반응변수가 2-차원 매트릭스이면 첫 열은 '성공 횟수' si를, 두 번째 열은 '실패 횟수' ai - si를 나타낸다. weights= 옵션이 필요 없다.

### a. GLM의 모형 적합도


```R
chall <- read.csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/challenger.csv')
chall <- tbl_df(chall)
glimpse(chall)
```

    Observations: 23
    Variables: 5
    $ o_ring_ct   <int> 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6...
    $ distress_ct <int> 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0...
    $ temperature <int> 66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53,...
    $ pressure    <int> 50, 50, 50, 50, 50, 50, 100, 100, 200, 200, 200, 200, 2...
    $ launch_id   <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...



```R
(chall_glm <-
    glm(cbind(distress_ct, o_ring_ct - distress_ct) ~
        temperature, data=chall, family='binomial'))

summary(chall_glm)
1 - pchisq(11.2, 1)
```


    
    Call:  glm(formula = cbind(distress_ct, o_ring_ct - distress_ct) ~ temperature, 
        family = "binomial", data = chall)
    
    Coefficients:
    (Intercept)  temperature  
         8.8169      -0.1795  
    
    Degrees of Freedom: 22 Total (i.e. Null);  21 Residual
    Null Deviance:	    20.71 
    Residual Deviance: 9.527 	AIC: 24.87



    
    Call:
    glm(formula = cbind(distress_ct, o_ring_ct - distress_ct) ~ temperature, 
        family = "binomial", data = chall)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.7526  -0.5533  -0.3388  -0.1901   1.5388  
    
    Coefficients:
                Estimate Std. Error z value Pr(>|z|)   
    (Intercept)  8.81692    3.60697   2.444  0.01451 * 
    temperature -0.17949    0.05822  -3.083  0.00205 **
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 20.706  on 22  degrees of freedom
    Residual deviance:  9.527  on 21  degrees of freedom
    AIC: 24.865
    
    Number of Fisher Scoring iterations: 6




0.000817973319994447


'deviance'는 선형 모형에서 잔차의 제곱합을 일반화한 것이다. <br/>
최대우도법에 의해 구해진 추정치에 대해 모형의 적합도를 우도함수(likelihood function)로 나타낸 것이다.
    - null deviance는 모형을 적합하기 전의 deviance
    - residual deviance는 모형을 적합한 후의 deviance
    - 이 둘사이가 충분히 줄었다면, 이 모형은 적합하다고 판단
    - 귀무가설, 즉 모형이 적합하지 않다는 가정 하에서는 두 deviance의 차이는 대략 chi-squared 분포를 따른다.
    - 지금 상황에서 두 deviance 차이는 20.7 - 9.52 = 11.2로, 자유도 1인 카이제곱 분포에서 이 값은 아주 큰, 나오기 어려운 값이다.
      (1 - pchisq(11.2, 1) = 0.0008...
    - 따라서 모형 적합의 p값은 실질적으로 0이며, 이 모형은 데이터를 의미있게 설명한다고 할 수 있다. 

### b. 로지스틱 모형 예측, 링크와 반응변수


```R
# 적합된 모형에서 temperature= 30일 때 '성공확률' 알아보기
predict(chall_glm, data.frame(temperature=30))

exp(3.45) / (exp(3.45) +1)
predict(chall_glm, data.frame(temperature=30), type='response')
```


<strong>1:</strong> 3.43215903839515



0.969231140642852



<strong>1:</strong> 0.968694607674951


첫 아웃풋이 [0, 1] 사이의 값이 아니다. predict() 함수의 default가 link로, 선형예측값 X\*beta hat을 출력하기 때문이다. <br/>
즉, beta hat 0 + beta hat 1 x = 8.82 - 0.179 * 30
<br/><br/>
* 선형예측값이 아닌 확률값을 얻으려면 이 값의 로지스틱 변환, 혹은 predict() 함수의 type 옵션을 'response'로 한다.

### c. GLM
- 지금 본 성공-실패 범주형 반응변수를 위한 '로지스틱 회귀모형'은 GLM 모형의 특수한 경우 중 하나.
- glm 모형은 이 외에도 다양한 distribution family와 link fn을 지원

패밀리 | 디폴트 링크함수 | 규봉: example
:-----|:-----|:-----
binomial | link = 'logit | 성공-실패 반응변수
gaussian | link = 'identity | 선형모형으로 lm()과 같다.
Gamma | link = 'inverse' | 양의 값을 가지는 수량형 반응변수<br/>(강우량, 전자제품이 고장날 때 까지 걸리는 시간, 점원이 4명의 손님을 응대하는데 걸리는 시간 등)
poisson | link = 'log' | 0, 1, 2... '개수'를 나타내는 반응변수<br/>(일일 교통사고 횟수, 단위지역의 연간 지진발생 횟수 등)
