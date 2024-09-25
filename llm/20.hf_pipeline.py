from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch

###### DistilBERT base uncased finetuned SST-2
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# classifier = pipeline("sentiment-analysis")
checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
revision="af0f99b"

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    revision = revision,
    clean_up_tokenization_spaces=True,
    max_length=128,           # 최대 토큰 길이 설정
    padding='max_length',     # 패딩 설정
    truncation=True           # 잘림 설정
)

model0 = AutoModel.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 파이프라인 생성
classifier = pipeline(
    "sentiment-analysis",
    model = model,
    revision = revision,
    tokenizer=tokenizer
)


###### ------ inference ------
raw_inputs = [
    "I've been waiting for a Huggingface course my whole life",
    "I hate this so much!"
]

# Tokenizer로 입력 데이터를 숫자로 변환
inputs = tokenizer(raw_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt"    # PyTorch 텐서 형식으로 반환
    )
print(inputs)

outputs0 = model0(**inputs)
print(outputs0)
print(outputs0.last_hidden_state.shape)


outputs = model(**inputs)
print(outputs)
print(outputs.logits.shape)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

model.config.id2label

result = classifier(raw_inputs)
print(result)