# Kaggle |  Feedback Prize - English Language Learning

<img width="1024" alt="Untitled" src="https://user-images.githubusercontent.com/103119868/218717980-5e82b0cf-944b-44d3-a5c9-4376e765f66f.png">

- 주최: Vanderbilt University, The Learning Agency Lab
- 주관: Kaggle
- 수행 기간: August 30, 2022 → November 29, 2022
- <a href="https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview" target="_blank" rel="noreferrer noopener" target="_blank" rel="noreferrer noopener">대회 홈페이지</a>

<br>
 
## 1. 프로젝트 개요

### 목표
- 8~12학년 영어 학습자(ELL)의 언어 능력을 평가하는 자동화된 피드백 시스템을 개선하기 위해 주어진 에세이에 대한 6가지 평가 지표 (cohesion, syntax, vocabulary, phraseology, grammar, conventions) 각각의 점수를 예측하는 것
<br>

### 데이터
- 제시된 데이터 세트(ELLIPSE corpus)는 8~12학년 영어 학습자(ELL)가 작성한 논증 에세이로 구성
- 에세이는 cohesion, syntax, vocabulary, phraseology, grammar, conventions의 6가지 분석 지표에 따라 채점되고 각 지표는 에세이 작성 능력의 구성 요소를 나타내며 점수가 높을수록 해당 척도의 높은 숙련도를 의미함

<br>

## 2. 주요 수행 내용

### 모델 앙상블
- 사전 학습된 10개 모델로 가중 추론(Inference)

### 후처리
- 평균 조정: 지표마다 추론한 결과에 train data에서의 지표 평균과 동일하도록 상수를 더해 추론 값의 평균과 train data의 평균값을 동일하게 조정
- 분포 조정: 지표마다 추론한 결과에 train data에서의 분포 비율과 동일하도록 상수를 곱해 train data의 분포와 동일하게 조정
- 점수 단위(0.5)로 조정: 지표마다 추론한 결과를 실제 점수 단위인 0.5 단위로 조정

<br>

## 3. 프로젝트를 통해 배운점

- 그동안은 데이터 또는 모델 위주의 접근법으로만 생각했다면 이번 프로젝트에서 후처리 방식으로 접근하는 방식을 시도해봤습니다.
- 무조건 어렵게 시도하는 것이 아니라, 할 수 있는 선에서 끝까지 포기하지 않는 것이 중요하다는 것을 느꼈습니다.

<br>

## 4. 최종 결과
- Silver medal 획득 (Top 5%, 130th /2654)