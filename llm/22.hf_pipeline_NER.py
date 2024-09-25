from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

######
# Named entity recognition

# ner = pipeline("ner")
checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
revision = "f2482bf"

# Tokenizer 객체 생성
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    revision=revision,
    clean_up_tokenization_spaces=True
)

# Model 객체 생성
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    revision=revision
)

# 파이프라인 생성
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer
)

# 예제 텍스트 NER
text = "Hugging Face is creating a tool that democratizes AI by making it accessible to everyone."
result = ner(text)
print(result)