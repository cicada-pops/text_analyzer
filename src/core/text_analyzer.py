
import pymorphy3
import re

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from typing import Dict, List, Any

from ..lib.utils import format_time
from ..lib.constants import CEFR_READING_SPEED, POS_MAPPING

class TextAnalyzer:
    """
    Класс для вычисления основных метрик текста.

    Атрибуты:
        text (str): Исходный текст, приведённый к нижнему регистру и очищенный от лишних пробелов.
        morph (pymorphy3.MorphAnalyzer): Морфологический анализатор для русского языка.
        tokenizer (PunktSentenceTokenizer): Токенизатор предложений, используется для разделения текста на предложения.
        stop_words (set[str]): Набор стоп-слов и знаков препинания для фильтрации.
        words (List[str]): Список токенизированных слов без знаков препинания, только русские слова.
        sentences (List[str]): Список предложений, полученных из исходного текста.
        word_count (int): Общее количество слов в тексте.
        character_count (int): Общее количество символов в тексте, включая пробелы.
        sentence_count (int): Количество предложений в тексте.
        long_word_count (int): Количество длинных слов (более 6 букв) в тексте.
        normalized_words (List[str]): Лемматизированный список слов, исключая стоп-слова.
        
    Методы:
        tokenize_words: Токенизирует текст на слова, исключая знаки препинания.
        tokenize_sentences: Токенизирует текст на предложения.
        calculate_reading_time: Расчет времени чтения по уровням CEFR.
        count_pos_tags: Подсчет частей речи в тексте.
        calculate_lix: Расчет индекса читаемости LIX.
        get_stats: Получение метрик удобочитаемости текста.
    
    Исключения:
        ValueError: Если в тексте нет слов, текст не строка или нет русских букв.
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

        :return: Список токенизированных слов.
        """
        
        return [word for word in word_tokenize(self.text, language="russian") if re.match(r'^[а-яё-]+$', word)]

    def tokenize_sentences(self) -> List[str]:
        """
        Токенизирует текст на предложения.

        :return: Список предложений.
        """
        
        return self.tokenizer.tokenize(self.text)

    def lemmatize_words(self, stopwords: List[str] = None) -> List[str]:
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

        :return: Словарь, где ключи — уровни CEFR, а значения — словари с временем чтения (study_time, skim_time) в минутах и секундах.
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

        :return: Словарь с частями речи и процентным соотношением.
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

        Индекс LIX = (число слов / число предложений) + 
        (100 * число длинных слов (более 6 букв) / число слов)

        :return: Значение индекса LIX.
        """

        return (self.word_count / self.sentence_count) + (100 * self.long_word_count / self.word_count)

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает детализированную статистику по тексту.

        :return: Словарь со следующими ключами:
            - "character_count" (int): Общее количество символов (включая пробелы).
            - "word_count" (int): Общее количество слов.
            - "sentence_count" (int): Количество предложений.
            - "pos_tags" (Dict[str, str]): Части речи с их процентным соотношением.
            - "reading_time" (Dict[str, Dict[str, str]]): Время чтения (study_time, skim_time) по уровням CEFR.
            - "lix_index" (float): Индекс читаемости LIX.
        """
        
        return {
            "character_count": self.character_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "pos_tags": self.count_pos_tags(),
            "reading_time": self.calculate_reading_time(),
            "lix_index": self.calculate_lix(),
        }