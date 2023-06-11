# DACON |  Speaker's Emotion Recognition AI Competition
<img width="1024" alt="banner" src="https://user-images.githubusercontent.com/103119868/229271264-5122b223-6556-4db3-bad7-5b1dd9ea7935.png">

- Host/Organizer: DACON
- Timeline: November 1, 2022 → December 12, 2022
- <a href="https://dacon.io/competitions/official/236027/overview/description" target="_blank" rel="noreferrer noopener">Competition site</a>

<br>

## 1. Project Overview
### Objective
- Classify the most appropriate emotion by looking at factors such as the speaker, utterance, and emotion in the conversation

### Data
- English dataset with Unique ID, Utterance, Speaker, Dialogue_ID, Target

<br>

## 2. Key action
- Deep learning framework: using PyTorch
- Share progress after experimentation based on different baselines for each team member
- Using Pre-trained Model in Hugging Face
    - `tae898/emoberta-large`, `tae898/emoberta-base`,<br>
    `roberta-large`, `roberta-base`,<br>
    `bhadresh-savani/roberta-base-emotion`
    - Attempted by applying several options, such as adjusting Learning rate, StratifiedKFold, Train/Valid Split ratio etc
- Experiment to check if Utterance is affected by context within the same Dialogue_ID
    - Learning after reconstruction by shuffling train data regardless of the order of conversation

<br>

## 3. What I learned from the project
- Importance of experiment management
- While running the large model, I felt the importance of resource (GPU) and efficiency.
- Of the total test data, 30% were used as Public scores (visible during the competition) and 70% as Private scores, and we learned that consistent scores for unseen data are important.

<br>

## 4. Final Result
- 5<sup>th</sup> place award / 259 teams  

    <img width="400" alt="coe2" src="https://user-images.githubusercontent.com/103119868/218729064-d90a2cf5-0d47-4be0-8d13-49a1543b6397.png">