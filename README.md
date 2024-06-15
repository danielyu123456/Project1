# DBN 알고리즘 (Deep Belief Network-심층 신뢰 신경망)

기계학습에 사용되는 그래프 생성 모델(generative graphical model)이다. 이는 알고리즘 그래프를 생성하는데, 한 개의 입력층에 각각의 잠재변수를 학습하기 위한 다층의 은닉층으로 구성되어 있다. 각층은 제한된 볼츠만 머신(RBM)으로 이루어져 있는데 계층간 연결은 있지만 계층내에선 연결이 없다는 점이 특징이다. 각 층에서 선행학습으로 가중치를 학습하고 초기에 임의로 설정된 가중치를 조정하는 것으로 오차를 줄일 수 있다.

<img width="607" alt="사진1" src="https://github.com/danielyu123456/Project1/assets/170755250/8eb4735f-fda6-4b02-9346-0380b87b26fa">

## DBN의 역사
DBN은 알고리즘 분야에서 꽤 역사적인 의미를 가지고 있다. 2006년 제프린 힌튼이 ‘A fast learning algorithm for deep belief nets’**라는 논문을 발표한다. 여기서 최초로 DBN이라는 새로운 알고리즘이 소개되었고 기존 알고리즘의 문제점들을 해결했다. 두가지 예로 사전 학습(Pretraining)을 통해 기울기 소실을 해결했고, 학습 도중 의도적으로 데이터를 누락시키는 ‘Drop out’으로 새로운 데이터를 소화시키지 못하던 기존 알고리즘의 문제를 해결한 것이다. 이것이 오늘날의 AI 딥러닝 전성시대를 열었다고 해도 과언이 아니다.

## DBN과 DNN 구분
두 모델 다 입력층과 출력층 사이에 은닉 신경망을 다수 쌓아놓은 구조인 점에서 구조적으로 유사하다. 하지만 DBN은 RBM만을 은닉층으로 쌓는다는 점이 차이점이라고 볼 수 있다. 따라서 DBN이 RBM을 통한 비지도학습으로 학습한다면 DNN은 결과값이 정해진 지도학습으로 학습한다. DBN은 입력을 재구성하고 백프로프(back-prop)로 미세 조정하기 위해 미리 처리되지만, DNN은 백프로프로 완전히 감독되는 교육을 받는다. DNN은 학습량이 많은 데이터를 처리하는데 유용하다면 DBN은 학습량이 적은 또는 교육 세트가 작은 경우에 유용하게 사용된다. 

## Deep Belief Network의 역사적 위치
현재 딥러닝 연구에서는 DBN이 중심적인 역할을 하고 있지 않는다. 2006년에 DBN 연구가 나왔을 당시, RBM을 이용한 비지도 학습은 다층 퍼셉트론 연구의 고질적인 문제들이었던 과적합 (Overfitting) 문제와 오차 기울기 소멸 문제를 해결할 수 있는 획기적인 방법이었으며, 딥러닝 연구의 발전이 다시 시작하는 데 큰 역할을 했었다.
하지만 그 이후로 지도 학습에서 앞서 말한 문제들을 해결하려는 연구들이 있었고, 그 결과 기존의 시그모이드 함수 대신에 사용 가능한 Rectified Linear Unit (ReLU) 함수와 학습 중에 무작위로 뉴런을 비활성화 함으로써 과적합 문제를 해결하는 Dropout Layer 등 현재 딥러닝에서 자주 사용되는 기술들이 개발되면서 DBN 연구에 대한 관심은 줄어들었다.


## 오차 기울기 소멸 문제 (Gradient Descent Vanishing)
심층 신경망은 입력층과 출력층으로 이루어진 퍼셉트론에서 더 복잡해진 형태로, 입력데이터가 보이지 않는 은닉층이 입력층과 출력층 사이에 생기게 된다.
순전파 (Feed Forward Propagation)는 입력한 데이터가 입력 > 은닉 > 출력 단계로 전달되며 변수가 계산되고 저장되는 방식이며, 최초 입력값에서 각 층에 존재하는 가중치와 연산하면서 최종 층까지 계산된 후, 실제 Label과의 오차를 계산하는 과정이다.
역전파 (Back Propagation)은 이 과정을 역으로 돌아가면서, 오차를 기반으로 중간 단계의 가중치를 업데이트하는 과정이다. 

<img width="452" alt="image" src="https://github.com/danielyu123456/Project1/assets/170755250/2cfbacaa-7591-43fc-924c-12acc41205e5">
기울기 소멸 문제 & 기울기 폭주 문제를 설명하는 그림

(출처: https://www.nomidl.com/deep-learning/what-is-vanishing-and-exploding-gradient-descent/)

그런데 초기 심층 신경망에서는 사람의 점진적 학습을 모방하기 위해 Sigmoid 함수를 사용했는데, 역전파 과정에서 활성화 함수 (Activation Function)으로 인한 문제가 생겼었다.
연쇄법칙 (Chain RUle)은 함성 함수의 미분은 함성 함수를 구성하는 각 함수의 미분 곱으로 나타낼 수 있다는 법칙인데, 역전파는 연쇄 법칙을 통해 기울기를 구하고, 거기에 학습률을 곱한 값을 기존 가중치에서 빼서 가중치를 업데이트하는 방식이다. 그런데 이 때 기울기를 구할 때 곱하는 값이 1보다 작으면 은닉층이 많을 수록 점점 더 많은 수가 곱해지며 가중치가 0으로 수렴하게 된다. (반대로 1보다 큰 수가 연쇄적으로 곱해지면 가중치가 발산하게 된다.)
그래서 망이 깊어질 수록 역전파 과정에서 전달되는 오차의 값은 0으로 수렴하며, 나중에는 가중치의 갱신이 거의 일어나지 않는 오차 기울기 소멸 문제 (Gradient Descent Vanishing)가 생기게 된다.
(반대 경우 문제는 오차 기울기 폭주 문제 (Gradient Descent Exploding)이라고 한다.)
이 문제들은 심층 신경망에서 은닉층의 층 수가 늘어날 수록 더 심해졌었고, 이런 심층 신경망 활용의 제한은 당시 컴퓨터의 성능 문제와 함께 2차 AI 겨울의 원인이 되었다.
이 상황을 해결하기 위해 개발된 방식 중 하나가 심층 신뢰 신경망 (Deep Belief Network)이다.


## RBM은 무엇인가?
DBN을 구성하는 각 층인 RBM을 간단하게 알아볼 필요가 있다. RBM은 Restricted Boltzmann Machines의 약자로 데이터가 입력되는 가시층(visible layer)과 가중치가 학습되는 은닉층으로 구성되어 있다. 같은 층의 노드들과는 전혀 연결되어 있지 않으며, 이 구조 때문에 ‘Restricted’ Boltzmann Machines라는 이름이 붙게 되었다.
가장 간단한 RBM 구조식 하나를 예로 들어보겠다.

<img width="607" alt="사진2" src="https://github.com/danielyu123456/Project1/assets/170755250/93a30188-3de7-4b11-bd7b-4f511090f685">


입력층에 입력 값 X를 넣는다 가정했을 때 은닉층의 가중치 W와 곱해진다. 
그리고 바이어스(b)를 더한 값인 (X*W+b)을 활성함수 f()에 넣으면 값 a가 나오는 방식이다.
##### 정리
    activation f((weight w * input x) + bias b ) = output a


## Deep Belief Network
<img width="561" alt="image" src="https://github.com/danielyu123456/Project1/assets/170755250/1b5ea14f-9551-4c54-b89f-e018270f9487">

DBN 층별 선훈련을 표현한 그림 (출처: https://rla020.tistory.com/40)

DBN은 층별 선훈련이 된 RDM을 쌓아올려나가는 심층 신경망이다.

여기에서 심층 신경망의 오차 기울기 소멸 문제 (Gradient Descent Vanishing)를 해결하는 방법은 층별 선훈련 (Layerwise Pre-Training)이다.
먼저 입력층(x)에 가까운 은닉층(h1)을 이용해 x - h1 -x를 비지도 사전 학습을 진행해 Input인 x를 복원한다. 그 다음 x - h1 사이의 가중치를 고정해놓고, 그 다음으로 h1 - h2 - h1 에 같은 과정을 거쳐 h1 을 복원한 뒤 h1 - h2 사이의 가중치를 고정하고, 이를 반복하며 원하는 개수만큼 볼츠만 머신을 쌓아서 전체 DBN 구조를 완성하는 것이다. 
이처럼 전 은닉층 계층에서 나온 결과값은 다음 은닉층 계층의 입력 데이터가 되며, 그 전 단계에서 사용된 가중치를 반복하면서, 마지막 은닉층까지 반복되는 구조인 것이다. 이런 층별 선훈련을 목표는 각 은닉층의 가중치를 가능한 만큼 목표값의 근사치로 만들려는 것이다. 
DBN의 초기 계층은 데이터의 기본 구조를 학습하고, 연속적인 계층은 더 추상적이고 높은 수준의 특징을 학습하게 된다. 또한 각 과정에서 은닉층이 단 하나이기 때문에 비지도 학습이 가능하다는 특징이 있다. 

이렇게 DBN 구조가 완성이 되면, 이후 목적에 맞춰서 미세 조정 (Fine-Tuning)을 한다. 그리고 역전파 (Back Propagation)과 순전파 (Feed-Forward) 알고리즘을 통해 최종 가중치를 계산한다. DBN은 기본적으로 비지도 학습을 사용하지만, 이 과정에서는 마지막 계층의 은닉층을 입력층으로 하고 출력층을 추가하여 지도 학습을 할 수도 있다. 


## DBN의 강・약점
### 강점
DBN의 가장 큰 강점은 역시 효율적인 비지도 학습 이다. 비지도 학습이란 알고리즘의 결과값을 미리 제공하지 않고 인공지능이 입력 단계에서 패턴과 상하관계를 찾아내는 머신 러닝의 일종이다. 
결과값을 미리 알 필요없이 입력 값으로만 학습을 수행하기 때문에 더 복잡한 정보를 처리할 수 있고, 특히 변칙이나 이상을 감지하는데 유용하다.

    A. 사전 학습: 각 층을 단계적으로 학습하여 초기 가중치를 효과적으로 설정할 수 있다.
    B. 특징 학습: 비지도 학습을 통해 데이터의 복잡한 특징을 추출하고 학습할 수 있다. 여러 층의 RBM을 통해 사전학습한후 어느정도 보정된 가중치를 계산하기 때문에 학습데이터가 부족한 데이터를 처리하는데 특히 유용하다.
    C. 쉬운 초기화: 비지도 학습을 통해 가중치를 효과적으로 초기화 할 수 있어, 심층 신경망 학습 중 기울기 소실의 발생을 최소화할 수 있다.
    D. 여러 층의 RBM으로 구성되어 있어 새로운 데이터를 추가하거나 구조 자체를 변경하기에 다른 알고리즘에 비해 상대적으로 쉽다.
### 약점
가장 큰 약점이라고 할 수 있는 것은 다수의 은닉층을 통한 비지도학습을 통해 학습하기 때문에 각 RBM층을 개별적으로 학습하는 과정에서 많은 시간이 소요된다는 점이다. 또한 더욱 정교한 결과값을 위해 RBM을 계속 쌓는 과정에서 알고리즘 모델이 지나치게 복잡해지고 오류가 발생할 수 있다.


## 비지도 학습은 무엇인가?
지도 학습은 결과값 즉 정답이 이미 정해져 있는 데이터를 가지고 기계 학습하는 것이다. 비지도 학습은 지도 학습의 반대이다. 결과값이 없는 데이터를 기계가 주도적으로 데이터 간의 패턴 또는 유사도를 학습한다. 
### 강점
•	간단한 데이터 준비 과정: 사전에 데이터를 분류하고 레이블을 지정할 필요가 없다. 
•	숨겨진 패턴(이상) 감지 탁월: 스스로 결과를 도출하는 과정에서 인간이 예상하지 못한 패턴이나 이상을 탐지한다.
•	대량의 데이터 활용 가능: 레이블이 없는 데이터를 처리할 수 있기 때문에 지도학습에 비해 많은 양의 데이터를 사용할 수 있다.
### 약점
•	평가 및 검증의 어려움: 정답이 없는 데이터를 학습하기 때문에 도출된 결과값을 무조건 정답이라 평가하기 어렵다.
•	정확도 문제: 지도 학습에 비해 정확도가 낮을 수 있다. 


## DBN 적용 사례
DBN은 비지도 학습을 통한 주도적 학습 하에 데이터 간의 패턴, 유사도, 이상을 감지하는데 탁월하여 여러 분야에서 적용되었다. 
### 이미지 인식
- 손 글씨 숫자 인식 (MNIST 데이터셋): MNIST 데이터셋에서 DBN을 사용하여 이미지의 특징을 추출하고, 분류 성능을 크게 향상시킬 수 있다.
- 얼굴 인식: 얼굴 이미지에서 중요한 특징을 추출하여 사람을 인식하는 데 사용할 수 있다.
### 음성 인식
- 자동 음성 인식(Automatic Speech Recognition, ASR): 음성 신호에서 특징적인 음향 특징을 추출하여 단어나 문장 목소리 등을 인식하는 데 DBN을 사용한다.
### 의료 데이터 분석
- 질병 예측: DBN은 환자의 복잡한 의료 데이터를 효율적으로 처리하고, 예측 정확도를 높이는 데 기여할 수 있습니다.
- 영상 진단: MRI나 CT 스캔 이미지에서 암 세포를 식별하는 데 DBN이 활용된다.
### 금융 데이터 분석
- 비정상 거개 및 사기 탐지: 금융 거래 데이터를 분석하여 이상 거래를 탐지하는 데 사용된다. 비정상적인 거래 내역의 미묘한 차이를 학습하여 사기 탐지의 정확도를 높일 수 있다.
- 주가 예측: 주식 시장의 거래량, 등락 패턴 등을 분석하여 주가를 예측하는데 사용되기도 한다.
### 추천 시스템
- 영화 추천: 사용자의 영화 선호도를 학습하여 영화 추천 시스템을 만들 수 있다. 


## Deep Belief Network 하이브리드
앞서 언급했듯이, DBN은 데이터 준비 과정이 간단하고, 숨겨진 패턴을 감지하는 데 탁월하고, 대량의 데이터를 활용 가능하다는 장점이 있지만, 평가 및 검증이 어렵고, 낮은 정확도를 가진다는 단점이 있다.
그래서 이 장점을 활용하면서도 단점을 극복하기 위해 몇몇 경우 다른 딥러닝 기법과 하이브리화 (Hybridization) 하는 경우도 있는데, 이와 관련된 두 연구를 소개하려 한다.
첫번째 연구는 DBN과 서포트 벡터 머신 (Support Vector Machine)을 하이브리드화할 수 있는 방법에 대한 연구이다. 접근법에서 두 가지 조합이 나왔는데, 하나는 SVM이 DBN보다 사전에 진행되는 Pre-Classification 접근법이고, 하나는 DBN이 SVM보다 사전에 진행되는 Post-Classification 접근법이다. 각 접근법의 효용성은 Precision, Accuracy, F-Measure로 측정되었으며, 두 접근법 모두 순수한 DBN 방법보다 3가지 측정 기준에서 발전하였고, Pre-Classification 방법이 더 뛰어나다고 측정되었다. 
이 연구는 SVM-DBN 하이브리드가 웹상에서 얻은 정보를  자동적으로 큐레이션과 집합하는 데 유용할 것으로 보았다.

두번째 연구는 DBN과 기계 학습 모델을 하이브리드화한 방법에 대한 연구이다. 인공위성 숲 사진의 이미지 세그멘테이션 (Image Segmentation)이 대상이며, 기존의 기계 학습 분류기가 픽셀과 텍스쳐 간 공간적 관계와 같은 Feature들을 추출하는 데 어려움을 겪었다는 문제인식에서 시작한다. 그래서 DBN을 기계 학습 모델에 적용해 인공위성 사진을 숲과 숲이 아닌 토지로 분류하는 능력을 키우려 한다. Random Forest, Linear Support Vector Machine, K-Nearest Neighbor, Linear Discriminant Analysis, Gaussian Naive Bayes 등의 분류기가 사용되었고, 성능은 Accuracy, Jaccard Score Index, Root Mean Square Error 등을 기준으로 측정되었다.
결과적으로 하이브리드 모형은 모든 분류기에서 성능이 증가했으며, 기존 주제에 대한 다른 연구들에 비해서도 뛰어났다는 결론에 도달했다.


## DBN 코드설명
- 라이브러리 가져오기: 데이터 처리(numpy, pandas), 기계 학습 모델(scikit-learn) 및 딥 러닝(tensorflow)을 위한 필수 Python 라이브러리를 가져옵니다.
- 데이터 세트 로드: 손으로 쓴 숫자의 28×28 픽셀 이미지 모음인 MNIST 데이터 세트는 scikit-learn의 fetch_openml을 사용하여 가져옵니다. 이 데이터 세트는 분류 알고리즘을 벤치마킹하는 데 일반적으로 사용됩니다.
- 전처리: 데이터 세트는 train_test_split을 사용하여 훈련 세트와 테스트 세트로 분할됩니다. 그런 다음 StandardScaler를 사용하여 데이터의 크기를 조정하여 정규화하며, 이는 종종 신경망의 성능을 향상시킵니다.
- RBM 레이어: 제한된 볼츠만 머신(RBM)은 지정된 수의 구성 요소와 학습 속도로 초기화됩니다. RBM은 입력을 재구성하여 데이터에서 패턴을 찾는 비지도 신경망입니다.
- 분류기 레이어: 로지스틱 회귀 분류기가 최종 예측 레이어로 선택됩니다. 로지스틱 회귀는 분류 작업을 위한 간단하면서도 효과적인 선형 모델입니다.
- DBN 파이프라인: RBM과 로지스틱 회귀 모델은 파이프라인으로 함께 연결됩니다. 이를 통해 RBM(특징 추출용)을 순차적으로 적용한 후 로지스틱 회귀(분류용)를 수행할 수 있습니다.
- 훈련: DBN을 구성하는 파이프라인은 전처리된 훈련 데이터(X_train_scaled)에 대해 훈련됩니다. RBM은 로지스틱 회귀 모델에서 숫자를 분류하는 데 사용되는 기능을 학습합니다.
- 평가: 마지막으로 훈련된 DBN의 성능이 테스트 세트에서 평가됩니다. 모델의 성능을 정량적으로 측정하기 위해 분류 정확도(dbn_score)가 인쇄됩니다.
<img width="653" alt="사진3" src="https://github.com/danielyu123456/Project1/assets/170755250/45e64e02-8307-4c81-a001-1494826570d7">

<img width="543" alt="사진4" src="https://github.com/danielyu123456/Project1/assets/170755250/53a715ba-5029-442a-9547-f68d6bc7a1d3">


출력에는 20회 반복을 통한 DBN(Deep Belief Network)의 훈련 진행 상황이 표시됩니다. 각 반복 중에 DBN의 RBM 부분은 데이터 구조를 이해하는 방법을 학습합니다. "의사 가능성"은 RBM이 데이터를 얼마나 잘 모델링하는지 추정하는 데 사용되는 척도입니다. 그러나 주어진 값은 음수이고 크기가 증가합니다. 이는 모델이 학습함에 따라 의사 가능성이 증가(또는 손실 감소)할 것으로 예상하므로 일반적으로 발생해서는 안 됩니다.

훈련 후 DBN은 약 21.14%의 분류 점수를 달성했습니다. 이 점수는 정확성을 측정하는 방법입니다. 이는 DBN이 테스트 데이터 세트에서 숫자 클래스를 21.14%의 시간 동안 정확하게 예측했음을 알려줍니다. 이는 매우 높은 점수가 아니며, 이는 모델이 이 작업에서 제대로 수행되지 않았음을 나타냅니다.







## 참고
- DBN의 역사 - [AI 이야기] 인공지능의 결정적 순간들, 세 번째 순서 (letr.ai) https://www.letr.ai/blog/story-20211112-1
- DBN과 DNN 구분 - 딥러닝(Deep Learning) 알고리즘 이해하기 : 네이버 블로그 (naver.com) 마지막 단락 https://m.blog.naver.com/sundooedu/221211368089
- Deep Belief Network의 역사적 위치 - https://jinseob2kim.github.io/deep_learning.html / https://www.tcpschool.com/deep2018/deep2018_deeplearning_history
- 오차 기울기 소멸 문제 - https://velog.io/@yunyoseob/Gradient-Vanishing-%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EC%86%8C%EC%8B%A4 / https://blog.naver.com/koreadeep/222600824716 / https://velog.io/@lighthouse97/%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EC%86%8C%EC%8B%A4-%EB%AC%B8%EC%A0%9C%EC%99%80-ReLU-%ED%95%A8%EC%88%98#:~:text=%EA%B8%B0%EC%9A%B8%EA%B8%B0%20%EC%86%8C%EC%8B%A4%20%EB%AC%B8%EC%A0%9C / https://casa-de-feel.tistory.com/36 / https://heytech.tistory.com/388 / https://kevinitcoding.tistory.com/entry/%EA%B0%9C%EB%85%90%EC%A0%81-%EC%A0%91%EA%B7%BC-%EC%9D%B8%EA%B3%B5-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98-%ED%95%99%EC%8A%B5%EA%B2%BD%EC%82%AC-%ED%95%98%EA%B0%95%EB%B2%95%EA%B3%BC-%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EC%86%8C%EC%8B%A4-%EB%AC%B8%EC%A0%9CVanishing-Gradient-Problem-%EC%99%84%EB%B2%BD#google_vignette
- RBM 정의,구조 - 초보자용 RBM(Restricted Boltzmann Machines) 튜토리얼 | by 안종찬 | Medium https://medium.com/@ahnchan2/초보자용-rbm-restricted-boltzmann-machines-튜토리얼-791ce740a2f0
- Deep Belief Network - https://rla020.tistory.com/40 / https://www.linkedin.com/pulse/unleashing-power-deep-belief-networks-unsupervised-m35df#:~:text=Greedy%20layer%2Dwise%20pretraining%3A%20This,the%20convergence%20of%20the%20network / https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220884698923&proxyReferer=https:%2F%2Fwww.google.com%2F / https://89douner.tistory.com/340 / https://ocw.snu.ac.kr/sites/default/files/NOTE/IML_Lecture%20(09).pdf / https://ebbnflow.tistory.com/165
- DBN 강점 - 코딩의 시작, TCP School https://www.tcpschool.com/deep2018/deep2018_deeplearning_algorithm
- 비지도학습 - 비지도 학습(Unsupervised Learning) 이해를 돕는 심플 가이드 (appier.com) https://www.appier.com/ko-kr/blog/a-simple-guide-to-unsupervised-learning
- DBN 적용사례 - 이미지 인식: Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. Neural Computation, 18(7), 1527-1554 / 음성 인식: Mohamed, A., Dahl, G. E., & Hinton, G. (2012). Acoustic modeling using deep belief networks. IEEE Transactions on Audio, Speech, and Language Processing, 20(1), 14-22 / 의료 데이터 분석: Miotto, R., Wang, F., Wang, S., Jiang, X., & Dudley, J. T. (2016). Deep learning for healthcare: review, opportunities and challenges. Briefings in Bioinformatics, 19(6), 1236-1246 / 금융 데이터 분석: Dixon, M. F., Klabjan, D., & Bang, J. H. (2016). Classification-based financial markets prediction using deep neural networks. Algorithmic Finance, 5(3-4), 1-10 / 추천 시스템: Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). Restricted Boltzmann machines for collaborative filtering. Proceedings of the 24th International Conference on Machine Learning, 791-798 / 추가내용: [Here’s Everything You Need To Know About Deep Belief Network (inc42.com)](https://inc42.com/glossary/deep-belief-networks/) / [Neural Network and Deep Belief Network | Baeldung on Computer Science](https://www.baeldung.com/cs/deep-belief-network)
- Deep Belief Network 하이브리드 - 첫번째 연구: https://aircconline.com/mlaij/V8N3/8321mlaij04.pdf / 두번째 연구: https://www.mdpi.com/2313-433X/10/6/132
