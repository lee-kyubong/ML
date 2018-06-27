
# 선형모델 선택 및 regularization

- 선형모델의 additive 상관관계: 한 독립변수가 종속변수에 미치는 계수만큼의 영향은 다른 설명변수 값에 독립<br/>
    ex) sales에 있어 tv와 radio 집행 시너지효과는 배제
- 최소제곱적합을 다른 적합절차로 대체해 단순선형모델을 개선할 수 있는 방법들
- n >> p: 최소제곱 추정치들이 낮은 분산을 갖는 경향 있고, test set에 대해서도 성능 좋을 것. 
- p > n: 더 이상 unique한 LSE가 존재하지 않는다. 분산이 무한대가 되어 최소제곱 방법은 전혀 사용할 수 없게 된다. <br/>
    추정된 계수들은 constraining, shrinking을 통해 무시해도 될 수준의 편향 증가 및 현저한 분산 감소를 시킬 수 있다. test set 예측 정확도 상승

일반 선형 모델 적합에 최소제곱 대신 사용할 대안<br/>
    a. subset selection: p개의 설명변수 중 선별. 줄어든 변수들의 서브셋에 최소제곱을 사용해 모델 적합 <br/>
    b. shrinkage: p개의 설명변수 모두를 포함하는 모델을 적합. 하지만 추정된 계수는 LSE에 비해 0 쪽으로 수축된다. 분산 줄이는 효과 <br/>
    c. Dimension Reduction: p개의 설명변수를 R^M으로 proj(M < p). 그 다음 M개의 설명변수로 최소제곱법을 수행해 선형회귀모델 적합 <br/>

a. AIC, BIC 낮은 것, adj R^2 높은 것 선택 or CV로 각 모델별 test error rate를 추정해 선택<br/>
![](https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-1-4614-7138-7_6/MediaObjects/978-1-4614-7138-7_6_Fig3_HTML.gif)

b. RIDGE(L2)
- It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
- It reduces the model complexity by coefficient shrinkage
- It uses L2 regularization technique.

c. LASSO(L1)
- We can see that as we increased the value of alpha, coefficients were approaching towards zero, but if you see in case of lasso, even at smaller alpha’s, our coefficients are reducing to absolute zeroes. Therefore, lasso selects the only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge.
- the lasso performs variable selection.
- We say that the lasso yields sparse models—that is, models that involve only a subset of the variables. As in ridge regression, selecting a good value of λ for the lasso is critical; we defer this discussion to Section 6.2.3, where we use cross-validation.

Now that you have a basic understanding of ridge and lasso regression, let’s think of an example where we have a large dataset, lets say it has 10,000 features. And we know that some of the independent features are correlated with other independent features. Then think, which regression would you use, Rigde or Lasso?

Let’s discuss it one by one. If we apply ridge regression to it, it will retain all of the features but will shrink the coefficients. _** But the problem is that model will still remain complex as there are 10,000 features, thus may lead to poor model performance. **_

Instead of ridge what if we apply lasso regression to this problem. The main problem with lasso regression is when we have correlated variables, it retains only one variable and sets other correlated variables to zero. _(규봉: Prof. Song says) ** That will possibly lead to some loss of information resulting in lower accuracy in our model. **_

Then what is the solution for this problem? Actually we have another type of regression, known as elastic net regression, which is basically a hybrid of ridge and lasso regression. So let’s try to understand it.

### L1과 L2의 우위 비교?

- **Neither ridge regression nor the lasso will universally dominate the other.** In general, one might expect the lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero. Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size. However, the number of predictors that is related to the response is never known a priori for real data sets. A technique such as cross-validation can be used in order to determine which approach is better on a particular data set.
As with ridge regression, when the least squares estimates have exces- sively high variance, the lasso solution can yield a reduction in variance at the expense of a small increase in bias, and consequently can gener- ate more accurate predictions. Unlike ridge regression, the lasso performs variable selection, and hence results in models that are easier to interpret.

- Cross-validation provides a simple way to tackle selecting the tuning parameter. We choose a grid of λ values, and compute the cross-validation error for each value of λ, as described in Chapter 5. We then select the tuning parameter value for which the cross-validation error is smallest. Finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter.

![L2](https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-1-4614-7138-7_6/MediaObjects/978-1-4614-7138-7_6_Fig12_HTML.gif)

- lamba가 0일 때, L1-L2 계수추정치는 OLS와 같다.
- 이 경우 ||beta hat of R||은 OLS의 norm과 같기에 비는 1이 된다.

![](https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-1-4614-7138-7_6/MediaObjects/978-1-4614-7138-7_6_Fig4_HTML.gif)
