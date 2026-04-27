import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import torch

# ---------------------------------------------------------
# 1. 데이터 전처리 (로컬 경로 적용)
# ---------------------------------------------------------
print("데이터를 불러오는 중입니다...")
# 파일 경로: 로컬 환경에 맞게 xlsx 파일로 로드합니다.
file_path = r"C:\unstructuredData\감성대화말뭉치(최종데이터)_Training.xlsx"

# 엑셀 파일 읽기 (openpyxl 패키지 필요)
df = pd.read_excel(file_path)

# 필요한 칼럼 추출 및 라벨 수치화
df = df[['사람문장1', '감정_대분류']].dropna()
df.columns = ['text', 'emotion']

def map_emotion(emo):
    if emo in ['기쁨']: return 0
    elif emo in ['당황']: return 1
    elif emo in ['불안', '상처', '슬픔']: return 2
    elif emo in ['분노']: return 3
    else: return 1

df['label'] = df['emotion'].apply(map_emotion)

# 프로토타입: 빠른 학습을 위해 1,000개만 샘플링
df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------
# 2. AI 모델 학습 (파인튜닝)
# ---------------------------------------------------------
print("AI 모델 세팅 중...")
dataset = Dataset.from_pandas(df_sample[['text', 'label']])
model_name = "beomi/KcELECTRA-base-v2022"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=10,
    save_strategy="no",
    # 로컬 CPU 환경 최적화를 위해 경고 메시지 끔 (최신 transformers 버전에 맞게 use_cpu 사용)
    use_cpu=not torch.cuda.is_available() 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("모델 학습을 시작합니다. (PC 사양에 따라 3~10분 정도 소요될 수 있습니다)")
trainer.train()
print("학습 완료!")

# ---------------------------------------------------------
# 3. 실시간 대화 추론 및 시계열 로직 테스트
# ---------------------------------------------------------
# 로컬 PC에 GPU가 있으면 0, 없으면 -1(CPU)로 자동 할당
device_id = 0 if torch.cuda.is_available() else -1 
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device_id)

score_map = {"LABEL_0": 1, "LABEL_1": 0, "LABEL_2": -1, "LABEL_3": -2}
emotion_name = {"LABEL_0": "기쁨", "LABEL_1": "당황/중립", "LABEL_2": "상처/슬픔", "LABEL_3": "분노"}

test_dialogue = [
    "안녕하세요, 배송이 안 와서 연락드렸어요.",
    "아니 어제 온다더니 왜 아직도 안 와요?",
    "진짜 너무하시네, 당장 환불해 주세요!",
    "장난합니까? 책임자 바꾸라고요!"
]

cumulative_score = 0

print("\n==================================================")
print(" 🚨 실시간 CS 위기 감지 테스트 시작")
print("==================================================")

for turn, text in enumerate(test_dialogue):
    result = nlp_pipeline(text)[0]
    label = result['label']
    
    turn_score = score_map[label]
    cumulative_score += turn_score
    
    print(f"\n[Turn {turn+1}] 고객: {text}")
    print(f" ➔ AI 분석: {emotion_name[label]} (정확도: {result['score']:.2f}) | 획득 점수: {turn_score}")
    print(f" ➔ 현재 누적 감정 점수: {cumulative_score}")
    
    if cumulative_score <= -4:
        print("\n 🚨 [경고] 폭발 임계점 돌파! 즉시 팀장급 상담원 교체 권고!")
        break
    elif cumulative_score <= -2:
        print("\n 🟡 [주의] 고객 감정 악화 중. 진정 멘트 사용 요망.")