## Dataset Info

- Input : 본문(content)의 문단(paragraph) 별 텍스트 및 질문  
- Output : 각 질문에 대한 답(paragraph 내에 있는 단어 일부, 각 질문 당 답은 하나) 


### train.json
- 데이터 형식: SQuAD(Stanford Question Answering Dataset) v2.0
- Content 3506개: 각 content는 여러 개의 paragraph로 구성됨
- content_id: Content에 부여된 ID
- title: Content 제목
- paragraphs: Content 내 Paragraph 개수
    - paragraph_id: Paragraph에 부여된 ID
    - context: Paragraph 본문 내용
    - qas: Paragraph 마다 하나의 Question이 지정됨
        - question_id: Question에 부여된 ID
        - question: Question 문장
        - answer
            - text
            - answer_start
        - is_impossible
            - 질문에 대한 답이 해당 paragraph에 있는 경우: `is_impossible:false`
            - 질문에 대한 답이 해당 paragraph에 없는 경우: `is_impossible:true`


### test.json
- 데이터 형식: SQuAD(Stanford Question Answering Dataset) v2.0
- Content 1038개: 각 content는 여러 개의 paragraph로 구성됨
- content_id: Content에 부여된 ID
- title: Content 제목
- paragraphs: Content 내 Paragraph 개수
    - paragraph_id: Paragraph에 부여된 ID
    - context: Paragraph 본문 내용
    - qas: Paragraph 마다 하나의 Question이 지정됨
        - question_id: Question에 부여된 ID
        - question: Question 문장


### sample_submission.csv
- question_id
- answer_text