"""
gpt_killer_ko.py

간단한 GPT 탐지기 (한국어 전용, 교육용 데모 버전)

<핵심 아이디어>

1) 통계적 패턴 분석
   - 문장/단어 길이, 단어 다양도(type-token ratio)
   - Hapax Legomenon Rate (한 번만 등장하는 단어 비율)
   - 기능어(function word) 비율 (한국어 접속사/기능어 위주)
   - 전형적인 AI-style 연결어/템플릿 표현 (한국어)
   - 반복되는 어구(bigram) 비율
   - 문장 길이 분산(균일한지 여부)

2) 문맥 분석(아주 단순한 버전)
   - 문장 길이 분포와 반복 패턴을 통해
     사람이 쓰기 어색한 "AI스러운" 패턴을 점수화

※ 실제 GPT 탐지기가 아니라, 통계적 패턴을 이용한 "대략적인 지표"를
   보여주는 교육용/연습용 도구이다.

<점수 체계>

- 내부적으로 0~10점(raw_score)으로 점수를 누적한 뒤
  0~100점(ai_score)으로 스케일링한다.
- 기본 판정 기준:
  - 70점 이상: AI(예: GPT)가 작성했을 가능성이 높음
  - 40~69점: 사람/AI 혼합 또는 애매
  - 39점 이하: 학생(사람)이 작성했을 가능성이 높음
"""

import sys
import re
from collections import Counter
from typing import List, Dict, Tuple


# -------------------------------
# 1. 텍스트 전처리 함수들
# -------------------------------

def split_sentences(text: str) -> List[str]:
    """
    매우 단순한 문장 분리:
    '.', '?', '!' + 줄바꿈 기준으로 문장을 나눈다.
    한국어 텍스트를 주로 대상으로 하지만, 마침표/물음표/느낌표 기준은 그대로 사용.
    """
    raw = re.split(r'[.!?]\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    """
    단어 토큰화:
    - 영문/숫자/한글만 남기고 소문자로 만든다.
    - 한국어 전용으로 사용하지만, 텍스트 안에 섞인 영어/숫자도 함께 토큰으로 취급.
    """
    tokens = re.findall(r"[A-Za-z가-힣0-9]+", text.lower())
    return tokens


# -------------------------------
# 2. 기본 통계 계산
# -------------------------------

def compute_basic_stats(text: str) -> Dict:
    sentences = split_sentences(text)
    tokens = tokenize(text)

    n_chars = len(text)
    n_sents = len(sentences)
    n_tokens = len(tokens)

    avg_sent_len = n_tokens / n_sents if n_sents else 0.0
    avg_token_len = (
        sum(len(t) for t in tokens) / n_tokens if n_tokens else 0.0
    )
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / n_tokens if n_tokens else 0.0  # type-token ratio

    # 한 번만 등장하는 단어 비율
    counter = Counter(tokens)
    hapax_count = sum(1 for _, c in counter.items() if c == 1)
    hapax_rate = hapax_count / n_tokens if n_tokens else 0.0

    return {
        "n_chars": n_chars,
        "n_sentences": n_sents,
        "n_tokens": n_tokens,
        "avg_sentence_length": avg_sent_len,
        "avg_token_length": avg_token_len,
        "type_token_ratio": ttr,
        "hapax_legomena_rate": hapax_rate,
        "sentences": sentences,
        "tokens": tokens,
        "counter": counter,
    }


# -------------------------------
# 3. 패턴/문맥 특징 계산 (한국어 전용)
# -------------------------------

def compute_pattern_features(
    text: str, tokens: List[str], sentences: List[str], counter: Counter
) -> Dict:
    n_tokens = len(tokens)
    most_common = counter.most_common(10)

    top1_freq = most_common[0][1] / n_tokens if (n_tokens and most_common) else 0.0
    top3_freq = (
        sum(f for _, f in most_common[:3]) / n_tokens if n_tokens else 0.0
    )

    # 전형적 연결어/결론 표현
    transition_phrases_ko = [
        "결론적으로",
        "요약하자면",
        "전반적으로",
        "종합하면",
        "정리하자면",
        "이 글에서는",
        "첫째",
        "둘째",
        "셋째",
        "마지막으로",
        "이러한 점에서",
        "이러한 측면에서",
        "중요성을",
        "관점에서 볼 때",
        "한편",
        "또한",
        "더불어",
        "따라서",
        "그러므로",
    ]

    tp_count = 0
    text_lower = text  # 한국어는 대소문자 차이가 없지만 변수만 통일
    for phrase in transition_phrases_ko:
        tp_count += text_lower.count(phrase)

    # 한국어 기능어/접속어
    function_words_ko = [
        "그리고",
        "그러나",
        "하지만",
        "또한",
        "그러므로",
        "따라서",
        "때문에",
        "그래서",
        "반면에",
        "한편",
        "혹은",
        "또는",
        "즉",
        "예를",
        "예를 들어",
    ]
    fw_count = sum(counter[w] for w in function_words_ko if w in counter)
    function_word_ratio = fw_count / n_tokens if n_tokens else 0.0

    # 1인칭 표현
    first_person = ["나", "내가", "제 생각에는", "제가", "우리가", "우리"]
    fp_count = 0
    for p in first_person:
        fp_count += text.count(p)
    fp_ratio = fp_count / (len(sentences) or 1)

    # 문장 길이 분산
    sent_lengths = [len(tokenize(s)) for s in sentences] or [0]
    mean_len = sum(sent_lengths) / len(sent_lengths)
    var_len = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)

    # 2단어 묶음 반복 비율
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counter = Counter(bigrams)
    repeated_bigrams = sum(1 for _, c in bigram_counter.items() if c >= 2)
    bigram_repeat_ratio = (
        repeated_bigrams / (len(bigram_counter) or 1)
        if bigram_counter
        else 0.0
    )

    return {
        "top1_freq": top1_freq,
        "top3_freq": top3_freq,
        "transition_phrase_count": tp_count,
        "function_word_ratio": function_word_ratio,
        "first_person_ratio_per_sentence": fp_ratio,
        "sentence_length_variance": var_len,
        "bigram_repeat_ratio": bigram_repeat_ratio,
        "most_common_tokens": most_common,
    }


# -------------------------------
# 4. AI 점수 계산(0~100점 스케일)
# -------------------------------

def score_ai(stats: Dict, patterns: Dict) -> Dict:
    raw_score = 0
    reasons: List[str] = []

    # 1) 평균 문장 길이
    if stats["avg_sentence_length"] >= 18:
        raw_score += 2
        reasons.append("문장 길이가 전반적으로 길고 복잡한 편입니다.")
    elif stats["avg_sentence_length"] >= 14:
        raw_score += 1
        reasons.append("문장 길이가 다소 긴 편입니다.")

    # 2) 단어 다양도
    if stats["type_token_ratio"] >= 0.45:
        raw_score += 1
        reasons.append("단어 다양도가 높은 편입니다.")

    # 3) 한 번만 등장하는 단어 비율
    if stats["n_tokens"] >= 100 and stats["hapax_legomena_rate"] <= 0.30:
        raw_score += 1
        reasons.append(
            "단 한 번만 등장하는 단어의 비율이 낮아 어휘가 다소 반복적인 경향을 보입니다."
        )

    # 4) 전형적인 연결어/템플릿 표현 개수
    if patterns["transition_phrase_count"] >= 5:
        raw_score += 2
        reasons.append(
            "전형적인 연결어/클리셰/결론 표현이 많이 사용되었습니다."
        )
    elif patterns["transition_phrase_count"] >= 2:
        raw_score += 1
        reasons.append(
            "여러 전형적인 연결어/클리셰/결론 표현이 사용되었습니다."
        )

    # 5) 기능어(function word) 비율
    if (
        patterns["function_word_ratio"] >= 0.30
        and stats["avg_sentence_length"] >= 14
    ):
        raw_score += 1
        reasons.append(
            "접속사/기능어 비율이 높고 문장 구조가 매우 정제된 편입니다."
        )

    # 6) 1인칭 표현 비율
    if patterns["first_person_ratio_per_sentence"] <= 0.1:
        raw_score += 1
        reasons.append("1인칭 표현이 거의 사용되지 않았습니다.")

    # 7) 반복되는 bigram 비율
    if patterns["bigram_repeat_ratio"] >= 0.2:
        raw_score += 1
        reasons.append("유사한 어구(두 단어 조합)가 여러 번 반복됩니다.")

    # 8) 문장 길이 분산
    if (
        patterns["sentence_length_variance"] <= 15
        and stats["avg_sentence_length"] >= 12
    ):
        raw_score += 1
        reasons.append("문장 길이가 비교적 균일합니다.")

    # ---------------------------
    # 100점 만점 스케일링
    # ---------------------------
    max_raw = 10
    ai_score_100 = round(raw_score / max_raw * 100)

    # 100점 기준 판정
    if ai_score_100 >= 70:
        label = "AI(예: GPT)가 작성했을 가능성이 높습니다."
    elif ai_score_100 >= 40:
        label = "AI와 사람이 섞였거나, 어느 쪽인지 애매합니다."
    else:
        label = "학생(사람)이 작성했을 가능성이 높습니다."

    return {
        "raw_score": raw_score,
        "ai_score": ai_score_100,
        "label": label,
        "reasons": reasons,
    }


# -------------------------------
# 5. 전체 분석 함수 + CLI
# -------------------------------

def analyze_text(text: str) -> Tuple[Dict, Dict, Dict]:
    stats = compute_basic_stats(text)
    patterns = compute_pattern_features(
        text,
        stats["tokens"],
        stats["sentences"],
        stats["counter"],
    )
    result = score_ai(stats, patterns)
    return stats, patterns, result


def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(
            "분석할 텍스트를 입력한 뒤, 입력을 종료하면 됩니다.\n"
            "(터미널/콘솔에서는 Ctrl+D(맥/리눅스) 또는 Ctrl+Z+Enter(윈도우) 사용)"
        )
        text = sys.stdin.read()

    stats, patterns, result = analyze_text(text)

    print("===== GPT Killer (한국어 전용) 결과 =====")
    print(f"- 글자 수: {stats['n_chars']}")
    print(f"- 문장 수: {stats['n_sentences']}")
    print(f"- 단어 수: {stats['n_tokens']}")
    print(f"- 평균 문장 길이(단어 기준): {stats['avg_sentence_length']:.2f}")
    print(f"- 평균 단어 길이: {stats['avg_token_length']:.2f}")
    print(f"- 단어 다양도(TTR): {stats['type_token_ratio']:.3f}")
    print(f"- Hapax 비율(HLR): {stats['hapax_legomena_rate']:.3f}")
    print()
    print(f"- 전형적 연결어/클리셰 개수: {patterns['transition_phrase_count']}")
    print(f"- 기능어 비율: {patterns['function_word_ratio']:.3f}")
    print(
        f"- 1인칭 표현 비율(문장당): "
        f"{patterns['first_person_ratio_per_sentence']:.3f}"
    )
    print(f"- 문장 길이 분산: {patterns['sentence_length_variance']:.2f}")
    print(f"- 반복 bigram 비율: {patterns['bigram_repeat_ratio']:.3f}")
    print()
    print(f"▶ 원시 점수(raw_score): {result['raw_score']} / 10")
    print(f"▶ AI 점수(ai_score): {result['ai_score']} / 100")
    print(f"▶ 최종 판단: {result['label']}")
    print()
    print("근거:")
    if result["reasons"]:
        for r in result["reasons"]:
            print(f" - {r}")
    else:
        print(" - 뚜렷하게 두드러지는 패턴이 감지되지 않았습니다.")

    print(
        "\n(주의) 이 도구는 통계적 패턴에 기반한 교육용 데모이며, "
        "실제 부정행위 판정용으로 사용할 수 없습니다."
    )


if __name__ == "__main__":
    main()
