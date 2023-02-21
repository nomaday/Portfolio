# DACON |  문장 유형 분류 AI 경진대회
![Untitled](https://user-images.githubusercontent.com/103119868/218712997-2ffe2c85-e0a3-495f-bbf0-279f0d040783.png)


- 주최: 성균관대학교
- 주관: DACON
- 수행 기간: December 12, 2022 → December 23, 2022
- <a href="https://dacon.io/competitions/official/236037/overview/description" target="_blank" rel="noreferrer noopener">대회 홈페이지</a>



## 1. 프로젝트 개요
### 목표
- 문장 유형 분류 AI 모델 개발  
    문장을 입력으로 받아 문장의 '유형', '시제', '극성', '확실성'을 분류하는 모델 생성


### 데이터
<img width="1208" alt="Untitled 1" src="https://user-images.githubusercontent.com/103119868/218712977-fda1a19b-3121-40bb-81e1-2c3d5f12dcaa.png">

- ID : 샘플 문장 별 고유 ID
- 문장 : 샘플 별 한개의 문장
- 유형 : 문장의 유형 (사실형, 추론형, 대화형, 예측형)
- 극성 : 문장의 극성 (긍정, 부정, 미정)
- 시제 : 문장의 시제 (과거, 현재, 미래)
- 확실성 : 문장의 확실성 (확실, 불확실)
- label : 문장 별 유형, 극성, 시제, 확실성에 대한 Class (총 72개 종류의 Class 존재)
    - 예시) 사실형-긍정-현재-확실
    

## 2. 주요 수행 내용
![Untitled 2](https://user-images.githubusercontent.com/103119868/218712993-2f988986-0f76-4e58-99f5-3f17d7bb9e5b.png)


- Multi-label classification을 다루면서 대회 평가기준인 Weighted f1 score에 적절한 모델이 무엇인지 관련 reference paper 찾기
    - <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9826728" target="_blank" rel="noreferrer noopener">BigBird</a> 모델이 적합한 것으로 보였음
- Hugging Face의 여러 PLM 적용 (`monologg/kobigbird-bert-base`, `klue/roberta-large`, `tunib/electra-ko-base`)
    - KFold, Batch size, Max length, Learning rate 등 여러 Parameter 조정해보면서 적용
- Team Notion 관리, 실험 관리 양식 작성 및 정리


## 3. 프로젝트를 통해 배운점
- 앙상블의 효과가 상당히 크다는 것을 느꼈습니다.
- 많은 옵션이 들어간다고 성능이 반드시 좋아지는 것은 아닌 점을 배웠습니다.
- 2주라는 기간동안 좀 더 다양한 시도를 해보았으면 좋았을 것 같습니다.


## 4. 최종 결과
- PUBLIC  : 15th / 333 teams
- **PRIVATE : 22th / 333 teams (Top 7%)**