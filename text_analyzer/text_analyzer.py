import re
from collections import defaultdict
import os

import nltk

nltk_data_path = os.path.expanduser('~/nltk_data')
flag_file = os.path.join(nltk_data_path, '.resources_downloaded')

if not os.path.exists(flag_file):
    for resource in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
    
    os.makedirs(nltk_data_path, exist_ok=True)
    with open(flag_file, 'w') as f:
        f.write('NLTK resources downloaded')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from typing import Dict, List, Any

import pymorphy3

from cefr_dictionary import a1, a2, b1, b2, c1, c2
from utils import format_time
from constants import CEFR_READING_SPEED, POS_MAPPING




class TextAnalyzer:
    """
    Класс для вычисления основных метрик текста и его лексического анализа.

    Атрибуты:
        text (str): Исходный текст, приведённый к нижнему регистру и очищенный от лишних пробелов.
        morph (pymorphy3.MorphAnalyzer): Морфологический анализатор для русского языка.
        tokenizer (PunktSentenceTokenizer): Токенизатор предложений для разделения текста на предложения.
        stop_words (set[str]): Набор стоп-слов для фильтрации.
        words (List[str]): Список токенизированных слов без знаков препинания (только русские буквы).
        normalized_words (List[str]): Лемматизированный список слов, исключая стоп-слова.
        sentences (List[str]): Список предложений, полученных из текста.
        word_count (int): Общее количество слов в тексте.
        character_count (int): Общее количество символов в тексте (включая пробелы).
        sentence_count (int): Количество предложений в тексте.
        long_word_count (int): Количество слов длиннее 6 символов.

    Новые метрики:
        - Количество уникальных слов и лексическое разнообразие.
        - Анализ покрытия текста по лексическим словарям (A1, A2, B1, B2, C1, C2, а также частотный словарь dictionary_500).
        - Определение уровня текста по системе ACTFL.
        - Выделение ключевых и самых полезных слов.

    Методы:
        tokenize_words: Токенизирует текст на слова, исключая знаки препинания.
        tokenize_sentences: Токенизирует текст на предложения.
        lemmatize_words: Лемматизирует слова из текста с исключением стоп-слов.
        calculate_reading_time: Рассчитывает время чтения для уровней CEFR.
        count_pos_tags: Подсчитывает части речи в тексте.
        calculate_lix: Рассчитывает индекс читаемости LIX.
        get_unique_word_count: Возвращает количество уникальных слов.
        get_lexical_diversity: Вычисляет отношение уникальных слов к общему числу слов.
        analyze_lexical_lists: Анализирует покрытие текста по заданным лексическим словарям.
        determine_actfl_level: Определяет уровень текста по системе ACTFL на основе лексического покрытия.
        get_key_words: Выделяет ключевые слова из базовых лексических словарей (A1 ∪ A2).
        get_most_useful_words: Выделяет слова, отсутствующие в базовых лексических словарях (A1 ∪ A2).
        get_stats: Возвращает полную статистику по тексту, объединяя базовые и новые метрики.
        get_full_analysis_text: Форматирует итоговый анализ в виде многострочного текста.

    Исключения:
        ValueError: Выбрасывается, если:
            - текст пустой или не является строкой,
            - текст не содержит русских букв,
            - в тексте меньше 5 слов.
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

        :return: Список токенизированных слов (только русские буквы).
        """
        return [word for word in word_tokenize(self.text, language="russian") if re.match(r'^[а-яё-]+$', word)]

    def tokenize_sentences(self) -> List[str]:
        """
        Токенизирует текст на предложения.

        :return: Список предложений.
        """
        return self.tokenizer.tokenize(self.text)

    def lemmatize_words(self) -> List[str]:
        """
        Лемматизирует слова из текста, исключая стоп-слова.

        :return: Список лемматизированных слов.
        """
        return [
            self.morph.parse(word)[0].normal_form for word in self.words
            if word not in self.stop_words
        ]

    def calculate_reading_time(self) -> Dict[str, Dict[str, str]]:
        """
        Рассчитывает время чтения текста для всех уровней CEFR в минутах и секундах.

        :return: Словарь, где ключи — уровни CEFR, а значения — словари с параметрами study_time и skim_time.
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
        Подсчитывает количество слов различных частей речи.

        :return: Словарь с названиями частей речи и их процентным соотношением.
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

        Формула:
            LIX = (число слов / число предложений) + (100 * число длинных слов (более 6 букв) / число слов)

        :return: Значение индекса LIX.
        """
        return (self.word_count / self.sentence_count) + (100 * self.long_word_count / self.word_count)

    # Новые методы

    def get_unique_word_count(self) -> int:
        """
        Возвращает количество уникальных слов (на основе исходного списка токенов).

        :return: Количество уникальных слов.
        """
        return len(set(self.words))

    def get_lexical_diversity(self) -> float:
        """
        Вычисляет лексическое разнообразие как отношение уникальных слов к общему числу слов.

        :return: Лексическое разнообразие, округленное до двух знаков после запятой.
        """
        return round(self.get_unique_word_count() / self.word_count, 2)

    def analyze_lexical_lists(self) -> Dict[str, Any]:
        """
        Анализирует покрытие текста по лексическим словарям.

        Для каждого словаря (A1, A2, B1, B2, C1, C2) определяется процент слов,
        присутствующих в этом словаре, а также список слов, отсутствующих в нём.
        Проводится также анализ по частотному словарю (dictionary_500).

        :return: Словарь с информацией по каждому лексическому списку.
        """
        lexical_analysis = {}
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
        # Анализ для частотного словаря dictionary_500
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

        Перебор словарей происходит от самого высокого уровня (C1) к базовому (A1).
        Если покрытие соответствующего словаря ≥ 50%, уровень текста определяется по этому словарю.

        :return: Строка с определённым уровнем (например, "Advanced Mid", "Intermediate High" и т.д.).
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

        :return: Отсортированный список ключевых слов.
        """
        basic_vocab = a1.union(a2)
        text_unique = set(self.words)
        key_words = sorted(list(text_unique.intersection(basic_vocab)))
        return key_words

    def get_most_useful_words(self) -> List[str]:
        """
        Выделяет «самые полезные слова» — слова, отсутствующие в базовых лексических словарях (A1 ∪ A2),
        то есть характеризующие более высокий уровень лексики.

        :return: Отсортированный список слов.
        """
        basic_vocab = a1.union(a2)
        text_unique = set(self.words)
        useful_words = sorted(list(text_unique - basic_vocab))
        return useful_words

     def get_full_analysis_text(self) -> str:
        """
        Форматирует итоговый анализ текста в виде многострочной строки.

        :return: Отформатированная строка с детальным анализом текста.
        """
        
        lines = [
            f"Уровень текста в системе ACTFL - {self.determine_actfl_level()}",
            f"Знаков с пробелами - {self.character_count}",
            f"Предложений - {self.sentence_count}",
            f"Слов - {self.word_count}",
            f"Уникальных слов - {self.get_unique_word_count()}",
            f"Лексическое разнообразие - {self.get_lexical_diversity()}"
        ]
        
        reading_times = self.calculate_reading_time()
        for cefr_level, times in reading_times.items():
            lines.append(f"Изучающее чтение: {times['study_time']}")
            lines.append(f"Просмотровое чтение: {times['skim_time']}")
        
        key_words = self.get_key_words()
        if key_words:
            lines.append("\nКлючевые слова:\n" + ", ".join(key_words))
        
        useful_words = self.get_most_useful_words()
        if useful_words:
            lines.append("\nСамые полезные слова:\n" + ", ".join(useful_words))
        
        lexical_lists = self.analyze_lexical_lists()
        for level, data in lexical_lists.items():
            lines.append(f"\nЛексический словарь {level} покрывает {data['coverage']}%")
            not_included = data.get("not_included", [])
            if not_included:
                lines.append(f"Не входит в лексический словарь {level}: " + ", ".join(not_included[:10]) + ("..." if len(not_included) > 10 else ""))
        
        return "\n".join(lines)

if __name__ == "__main__":
    import sys
    
    input_text = sys.stdin.read().strip()
    
    analyzer = TextAnalyzer(input_text)
    print(analyzer.get_full_analysis_text())
