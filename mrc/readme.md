# AI CONNECT |  YearDream Competition
<img width="960" alt="Picture2" src="https://github.com/nomaday/Portfolio/assets/103119868/d4b9bfaf-4b29-4ef6-8f22-93ec9f92d2f8">


- Host: Korea SMEs and Startups Agency
- Organizer: AI CONNECT (MINDS AND COMPANY)
- Timeline: November 14, 2022 → November 28, 2022
- <a href="https://aiconnect.kr/competition/detail/217" target="_blank">Competition site</a>

<br>

## 1. Project Overview
- Machine reading comprehension for efficiency of book data search
- Natural Language Processing (NLP) | open question | Accuracy

#### Problem definition
- Machine Reading Comprehension to find answers to questions in given passages

<br>

## 2. Key action
- Investigation of information related to machine reading comprehension
- Use Pre-trained model
    - Apply `monologg/koelectra-base-v3-discriminator`  model
    - Apply `KLUE-RoBERTa-large` model  
        *(The final score confirmed after the competition was 0.624331, the highest)*
- Use HuggingFace library
    - `ElectraForQuestionAnswering` `ElectraTokenizerFast`
    - `AutoModelForQuestionAnswering`, `AutoTokenizer`
- Parameters adjustment
    - Learning rate, Batch size, Epoch
    - CosineAnnealingLR

<br>

## 3. Final Result
- (Final LB) 2<sup>nd</sup> / 139 people 
- (Public LB) 1<sup>st</sup> / 139 people