from transformers import pipeline
from transformers import AutoTokenizer

######
# sshleifer/distilbart-cnn-12-6

# summarizer = pipeline("summarization")
# 모델 이름과 버전 지정
model_name = "sshleifer/distilbart-cnn-12-6"
revision = "a4f8f3e"
# Tokenizer 객체 생성
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=revision,
    clean_up_tokenization_spaces=True
)
# 파이프라인 생성
summarizer = pipeline(
    "summarization",
    model=model_name,
    revision=revision,
    tokenizer=tokenizer
)

# 예제 텍스트 요약
text = """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
result = summarizer(text)
print(result)

# [{'summary_text':
# ' The number of engineering graduates in the United States has declined in recent years .
# China and India graduate six and eight times as many traditional engineers as the U.S. does .
# Rapidly developing economies such as China continue to encourage and advance
# the teaching of engineering .
# There are declining offerings in engineering subjects dealing with infrastructure,
# infrastructure, the environment, and related issues .'}]