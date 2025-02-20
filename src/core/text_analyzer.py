import re
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from typing import Dict, List, Any

import pymorphy3

from ..lib.cefr_dictionary import a1, a2, b1, b2, c1, c2, dictionary_500
from ..lib.utils import format_time
from ..lib.constants import CEFR_READING_SPEED, POS_MAPPING


class TextAnalyzer:
    """
    Класс для вычисления основных метрик текста и его лексического анализа.

    Новые метрики:
      - количество уникальных слов и лексическое разнообразие;
      - анализ покрытия текста по лексическим словарям (A1, A2, B1, B2, C1, C2, а также частотный словарь dictionary_500);
      - определение уровня текста по системе ACTFL;
      - выделение ключевых и самых полезных слов.
    """

    def __init__(self, text: str) -> None:
        if not text or not isinstance(text, str):
            raise ValueError("Текст должен быть непустой строкой.")

        if not any(char.isalpha() and char.islower() for char in text):
            raise ValueError("Текст должен содержать русские буквы.")

        self.morph = pymorphy3.MorphAnalyzer()
        self.tokenizer = PunktSentenceTokenizer()
        self.text = text.strip().lower()

        self.words = self.tokenize_words()

        if len(self.words) < 5:
            raise ValueError("Текст должен содержать не менее 5 русских слов.")

        self.stop_words = set(stopwords.words('russian'))
        self.normalized_words = self.lemmatize_words()
        self.sentences = self.tokenize_sentences()

        self.word_count = len(self.words)
        self.character_count = len(self.text)
        self.sentence_count = len(self.sentences)
        self.long_word_count = sum(1 for word in self.words if len(word) > 6)

    def tokenize_words(self) -> List[str]:

        """
        Токенизирует текст на слова, исключая знаки препинания.
        """

        return [word for word in word_tokenize(self.text, language="russian") if re.match(r'^[а-яё-]+$', word)]

    def tokenize_sentences(self) -> List[str]:
        """
        Токенизирует текст на предложения.
        """
        return self.tokenizer.tokenize(self.text)

    def lemmatize_words(self) -> List[str]:
        """
        Лемматизирует слова из текста, исключая стоп-слова.
        """
        return [
            self.morph.parse(word)[0].normal_form for word in self.words
            if word not in self.stop_words
        ]

    def calculate_reading_time(self) -> Dict[str, Dict[str, str]]:
        """
        Рассчитывает время чтения текста для всех уровней CEFR в минутах и секундах.
        """
        reading_times = {}
        for cefr_level, speeds in CEFR_READING_SPEED.items():
            study_speed = speeds["study"]
            skim_speed = speeds["skim"]

            study_time = self.word_count / study_speed
            skim_time = self.word_count / skim_speed

            reading_times[cefr_level] = {
                "study_time": format_time(study_time),
                "skim_time": format_time(skim_time),
            }

        return reading_times

    def count_pos_tags(self) -> Dict[str, str]:
        """
        Подсчитывает количество слов разных частей речи.
        """
        pos_count = defaultdict(int)
        for word in self.words:
            tag = self.morph.parse(word)[0].tag
            for pos, name in POS_MAPPING.items():
                if pos in tag:
                    pos_count[name] += 1
                    break
        return {pos: f"{count} ({(count / self.word_count) * 100:.2f}%)"
                for pos, count in pos_count.items()}

    def calculate_lix(self) -> float:
        """
        Рассчитывает индекс читаемости LIX.
        """
        return (self.word_count / self.sentence_count) + (100 * self.long_word_count / self.word_count)

    # Новые методы

    def get_unique_word_count(self) -> int:
        """
        Возвращает количество уникальных слов (на основе исходного токенизированного списка).
        """
        return len(set(self.words))

    def get_lexical_diversity(self) -> float:
        """
        Вычисляет лексическое разнообразие: отношение количества уникальных слов к общему числу слов.
        """
        return round(self.get_unique_word_count() / self.word_count, 2)

    def analyze_lexical_lists(self) -> Dict[str, Any]:
        """
        Анализирует покрытие текста по лексическим словарям.

        Для каждого словаря (A1, A2, B1, B2, C1, C2) определяется процент слов,
        присутствующих в словаре, и список слов, которых в нём нет.
        Также проводится анализ по частотному словарю (dictionary_500).
        """
        lexical_analysis = {}
        # Словарь лексических списков уровня
        lexical_lists = {
            "A1": a1,
            "A2": a2,
            "B1": b1,
            "B2": b2,
            "C1": c1,
            "C2": c2,
        }
        text_unique = set(self.words)
        for level, word_set in lexical_lists.items():
            intersection = text_unique.intersection(word_set)
            coverage = (len(intersection) / len(text_unique)) * 100 if text_unique else 0
            not_included = sorted(list(text_unique - word_set))
            lexical_analysis[level] = {
                "coverage": round(coverage),
                "not_included": not_included
            }
        # Анализ частотного словаря dictionary_500
        intersection = text_unique.intersection(dictionary_500)
        coverage = (len(intersection) / len(text_unique)) * 100 if text_unique else 0
        rare_words = sorted(list(text_unique - dictionary_500))
        lexical_analysis["dictionary_500"] = {
            "coverage": round(coverage),
            "rare_words": rare_words
        }
        return lexical_analysis

    def determine_actfl_level(self) -> str:
        """
        Определяет уровень текста по системе ACTFL на основе покрытия лексических словарей.

        Проходится по словарям от самого высокого (C1) к базовому (A1). Если покрытие
        соответствующего словаря ≥ 50%, уровень определяется по этому словарю.
        """
        analysis = self.analyze_lexical_lists()
        if analysis.get("C1", {}).get("coverage", 0) >= 50:
            return "Advanced Mid"
        elif analysis.get("B2", {}).get("coverage", 0) >= 50:
            return "Advanced Low"
        elif analysis.get("B1", {}).get("coverage", 0) >= 50:
            return "Intermediate High"
        elif analysis.get("A2", {}).get("coverage", 0) >= 50:
            return "Intermediate Mid"
        elif analysis.get("A1", {}).get("coverage", 0) >= 50:
            return "Novice High"
        else:
            return "Below Novice"

    def get_key_words(self) -> List[str]:
        """
        Выделяет «ключевые слова» — слова, присутствующие в базовых лексических словарях (A1 ∪ A2).
        """
        basic_vocab = a1.union(a2)
        text_unique = set(self.words)
        key_words = sorted(list(text_unique.intersection(basic_vocab)))
        return key_words

    def get_most_useful_words(self) -> List[str]:
        """
        Выделяет «самые полезные слова» — слова, отсутствующие в базовых лексических словарях (A1 ∪ A2),
        то есть характеризующие более высокий уровень лексики.
        """
        basic_vocab = a1.union(a2)
        text_unique = set(self.words)
        useful_words = sorted(list(text_unique - basic_vocab))
        return useful_words

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает детализированную статистику по тексту, включая новые метрики.
        """
        return {
            "actfl_level": self.determine_actfl_level(),
            "character_count": self.character_count,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "unique_word_count": self.get_unique_word_count(),
            "lexical_diversity": self.get_lexical_diversity(),
            "key_words": self.get_key_words(),
            "most_useful_words": self.get_most_useful_words(),
            "lexical_lists": self.analyze_lexical_lists(),
            "lix_index": self.calculate_lix(),
            "reading_time": self.calculate_reading_time(),
            "pos_tags": self.count_pos_tags(),
        }

    def get_full_analysis_text(self) -> str:
        """
        Форматирует итоговый анализ в виде многострочной строки, аналогичной примеру:

        Уровень текста в системе ACTFL - Advanced Mid
        Знаков с пробелами - 288
        Предложений - 1
        Слов - 36
        Уникальных слов - 26
        Лексическое разнообразие - 0.72

        Ключевые слова:
        текст, рассчитать, слово

        Самые полезные слова:
        разнообразие, лексический, сложность, коэффициент, ключевой

        Лексический список A1 покрывает 44%
        Не входит в лексический список A1: текстометр, определять, ...
        ...
        Частотный словарь dictionary_500 покрывает 0%
        Редкие слова:
        """
        stats = self.get_stats()
        lines = []
        lines.append(f"Уровень текста в системе ACTFL -\t{stats['actfl_level']}")
        lines.append(f"Знаков с пробелами -\t{stats['character_count']}")
        lines.append(f"Предложений -\t{stats['sentence_count']}")
        lines.append(f"Слов -\t{stats['word_count']}")
        lines.append(f"Уникальных слов -\t{stats['unique_word_count']}")
        lines.append(f"Лексическое разнообразие -\t{stats['lexical_diversity']}")
        lines.append("")
        lines.append("Ключевые слова:\t" + ", ".join(stats["key_words"]))
        lines.append("")
        lines.append("Самые полезные слова:\t" + ", ".join(stats["most_useful_words"]))
        lines.append("")
        for level, data in stats["lexical_lists"].items():
            if level == "dictionary_500":
                lines.append(f"Частотный словарь {level} покрывает\t{data['coverage']}%")
                lines.append("Редкие слова:\t" + ", ".join(data.get("rare_words", [])))
            else:
                lines.append(f"Лексический словарь {level} покрывает\t{data['coverage']}%")
                lines.append(f"Не входит в лексический словарь {level}:\t" + ", ".join(data.get("not_included", [])))
        return "\n".join(lines)


if __name__ == "__main__":
    sample_text = (
        "Компания Microsoft объявила о создании первого в мире квантового чипа Majorana 1 на топологических проводниках."
        "В частности, топопроводник способен создавать новое состояние материи, так что в недалеком будущем появится квантовый компьютер с миллионом кубитов."
    )
    analyzer = TextAnalyzer(sample_text)
    print(analyzer.get_full_analysis_text())
