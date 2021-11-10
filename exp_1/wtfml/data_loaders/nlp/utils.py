import json
import os
import re

import emoji
import mojimoji
import neologdn

working_direction = os.getcwd()

emoji_json_path = os.path.join(working_direction, "emoji/emoji_ja.json")
json_open = open(emoji_json_path)
emoji_dict = json.load(json_open)


def clean_sentence(sentence: str) -> str:
    """
    Bertに入れる前にtextに行う前処理

    Args:
        sentence (str): [description]

    Returns:
        str: [description]
    """
    sentence = re.sub(r"<[^>]*?>", "", sentence)  # タグ除外
    sentence = mojimoji.zen_to_han(sentence, kana=False)
    sentence = neologdn.normalize(sentence)
    sentence = re.sub(
        r'[!"#$%&\'\\\\()*+,\-./:;<=>?@\[\]\^\_\`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠？！｀＋￥％︰-＠]。、♪',
        " ",
        sentence,
    )  # 記号
    sentence = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", sentence)
    sentence = re.sub(r"[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+", " ", sentence)

    # ！小川：絵文字のエラーの解消（コメントアウトされていたものは削除しました。）！
    sentence = "".join(
        [
            "絵文字" + emoji_dict.get(c, {"short_name": ""}).get("short_name", "")
            if c in emoji.UNICODE_EMOJI["en"]
            else c
            for c in sentence
        ]
    )

    return sentence
