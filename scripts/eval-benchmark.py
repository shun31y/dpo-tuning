import sys

sys.path.append("..")
import csv
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from src import settings
from pathlib import Path
import fire


openai.api_key = settings.OPENAI_API_KEY  # OpenAI APIキーを設定してください


# https://beta.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt4eval(pred, input_text, output_text, eval_aspect):
    prompt = f"""あなたは採点者です。

    問題, 正解例, 採点基準, 回答 が与えられます。

    採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

    # 問題

    {input_text}

    # 正解例

    {output_text}

    # 採点基準

    基本的な採点基準

    - 1点: 誤っている、 指示に従えていない
    - 2点: 誤っているが、方向性は合っている
    - 3点: 部分的に誤っている、 部分的に合っている  
    - 4点: 合っている
    - 5点: 役に立つ

    基本的な減点項目

    - 不自然な日本語: -1点
    - 部分的に事実と異なる内容を述べている: -1点 
    - 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

    問題固有の採点基準

    {eval_aspect}

    # 回答

    {pred}

    """

    response = completion_with_backoff(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    gpt4score = response.choices[0].message.content
    try:
        gpt4score = int(gpt4score)
    except ValueError:
        gpt4score = None

    return gpt4score


def gpt4eval_with_csv_file(input_csv: str, output_csv: str) -> None:
    # 暗黙的にdata/result以下にinputを期待している
    input_file = Path(__file__).parent.parent / "data" / "result" / input_csv
    output_file = Path(__file__).parent.parent / "data" / "result" / output_csv

    with open(input_file, "r", encoding="utf-8") as csvfile, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ["score"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            """
            csvファイルは以下のカラムを持つ
            'input' : 入力テキスト
            'output' : 出力テキスト
            'eval_aspect' : 評価観点
            'pred' : 予測結果
            """
            input_text = row["input"]
            output_text = row["output"]
            eval_aspect = row["eval_aspect"]
            pred = row["pred"]

            score = gpt4eval(pred, input_text, output_text, eval_aspect)
            row["score"] = score

            writer.writerow(row)

    print(f"GPT-4による自動評価が完了しました。結果は {output_file} に出力されました。")


if __name__ == "__main__":
    fire.Fire(gpt4eval_with_csv_file)
