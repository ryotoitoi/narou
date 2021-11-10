import re

def clean_sentence(sentence: str) -> str:
    
    sentece = re.sub(r"<[^>]*?", "", sentence)
    sentence = re.sub(
        r'[!"#$%&\'\\\\()*+,\-./:;<=>?@\[\]\^\_\`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠？！｀＋￥％︰-＠]。、♪',
        " ",
        sentence,
    )  # 記号

    return sentence