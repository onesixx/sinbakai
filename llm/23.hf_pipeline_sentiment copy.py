from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

###### DistilBERT base uncased finetuned SST-2
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# classifier = pipeline("sentiment-analysis")
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
revision="af0f99b"
''' (model이 이해할수 있는 형식으로) 텍스트 data를 숫자 vector로 변환하는 도구
"Hello, world"
-(분할)-> ["Hello", ",", "world", "!"]
-(숫자화)-> [101, 100, 102, 103]
-(패딩/잘림)-> [101, 100, 102, 103, 0, 0, 0, 0, ...
'''
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision = revision,
    clean_up_tokenization_spaces=True,
    max_length=128,           # 최대 토큰 길이 설정
    padding='max_length',     # 패딩 설정
    truncation=True           # 잘림 설정
)

# 파이프라인 생성
classifier = pipeline(
    "sentiment-analysis",
    model = model_name,
    revision = revision,
    tokenizer=tokenizer
)

result = classifier("I've been waiting for a Huggingface course my whole life")
print(result)

result_multi = classifier([
    "I've been waiting for a Huggingface course my whole life",
    "I hate this so much!"
])
print(result_multi)