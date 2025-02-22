import re
from collections import defaultdict
import os
from freq_dictionary import freq_dict
from typing import Tuple
from rank_bm25 import BM25Okapi
import nltk
import g4f
from g4f.Provider import Blackbox

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
from constants import CEFR_READING_SPEED, POS_MAPPING, ACTFL_TO_CEFR


class TextAnalyzer:
    """
    Класс для выполнения лексического анализа и расчёта метрик текста.

    Основные возможности:
      - Токенизация текста на слова и предложения.
      - Лемматизация с использованием pymorphy3 и фильтрация стоп-слов.
      - Расчёт статистических показателей (количество слов, символов, предложений, уникальных слов, индекс читаемости LIX).
      - Анализ лексического покрытия по словарям уровней CEFR (A1, A2, B1, B2, C1, C2) и по частотному словарю (freq_dict).
      - Определение уровня текста по системе ACTFL на основе покрытия лексических списков.
      - Выделение ключевых слов на основе TF/IDF: вычисляется отношение числа появлений слова в тексте к его частоте по Национальному корпусу (с корректирующим коэффициентом).
      - Выделение «полезных слов» – тех, которые отсутствуют в базовых словарях (A1 ∪ A2) – с сортировкой по TF/IDF.

    Атрибуты:
      text (str): Исходный текст в нижнем регистре.
      morph (pymorphy3.MorphAnalyzer): Морфологический анализатор для русского языка.
      tokenizer (PunktSentenceTokenizer): Токенизатор предложений.
      stop_words (set[str]): Множество стоп-слов.
      words (List[str]): Токенизированные слова.
      normalized_words (List[str]): Лемматизированные слова (без стоп-слов).
      sentences (List[str]): Список предложений.
      word_count (int): Общее число слов.
      character_count (int): Количество символов (с пробелами).
      sentence_count (int): Количество предложений.
      long_word_count (int): Число слов длиннее 6 символов.

    Методы:
      tokenize_words: Токенизация текста на слова.
      tokenize_sentences: Разбиение текста на предложения.
      lemmatize_words: Лемматизация с исключением стоп-слов.
      calculate_reading_time: Расчёт времени чтения по уровням CEFR.
      count_pos_tags: Подсчёт частей речи.
      calculate_lix: Вычисление индекса читаемости LIX.
      get_unique_word_count: Подсчёт уникальных слов.
      get_lexical_diversity: Расчёт лексического разнообразия.
      get_words: Извлечение слов из предложения.
      get_syllables_count: Подсчёт количества слогов в слове.
      get_average_sentence_length: Вычисление средней длины предложения.
      get_average_word_length: Определение средней длины слова.
      get_average_syllables_word: Расчёт среднего количества слогов на слово.
      get_flesh_index: Вычисление индекса удобочитаемости Флеша.
      get_bm25_scores: Вычисление BM25-оценок для предложений по заданному запросу.
      analyze_lexical_lists: Анализ покрытия текста по словарям (CEFR и freq_dict).
      determine_actfl_level: Определение уровня текста по системе ACTFL.
      get_key_words: Выделение ключевых слов по метрике TF/IDF.
      get_most_useful_words: Выделение полезных слов (отсутствующих в A1 ∪ A2) с использованием TF/IDF.
      get_full_analysis_text: Форматирование полного анализа в виде многострочного текста.

    Исключения:
      ValueError: Выбрасывается, если текст пустой, не является строкой, не содержит русских букв или содержит менее 5 слов.
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

    def calculate_reading_time(self, cefr_level: str) -> Dict[str, str]:
        """
        Рассчитывает время чтения текста для заданного уровня CEFR в минутах и секундах.

        :param cefr_level: Уровень CEFR, для которого требуется рассчитать время чтения.
        :return: Словарь с параметрами study_time и skim_time для указанного уровня.
        """
        if cefr_level not in CEFR_READING_SPEED:
            raise ValueError(f"Неизвестный уровень CEFR: {cefr_level}")

        speeds = CEFR_READING_SPEED[cefr_level]
        study_speed = speeds["study"]
        skim_speed = speeds["skim"]

        study_time = max(self.word_count / study_speed, 1)
        skim_time = max(self.word_count / skim_speed, 1)

        return {
            "study_time": format_time(study_time),
            "skim_time": format_time(skim_time),
        }

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
    
    def get_words_in_sentence(self, sentence) -> List[str]:
        """
        Извлекает слова из предложения.

        :param sentence: Входное предложение.
        :return: Список слов в предложении.
        """
        return re.findall(r'\b\w+\b', sentence)
    
    def get_syllables_count(self, word) -> int:
        """
        Подсчитывает количество слогов в слове.

        :param word: Входное слово.
        :return: Количество слогов в слове.
        """
        return len(re.findall(r'[аеёиоуыэюя]', word, re.IGNORECASE))
    
    def get_average_sentence_length(self) -> float:
        """
        Рассчитывает среднюю длину предложения (количество слов в среднем предложении).

        :return: Средняя длина предложения.
        """
        sentences = self.tokenize_sentences()
        count_in_sentences = [len(self.get_words_in_sentence(sentence)) for sentence in sentences]
        return sum(count_in_sentences) / len(sentences) if len(sentences) != 0 else 0
    
    def get_average_word_lenght(self) -> float:
        """
        Рассчитывает среднюю длину слова в тексте.

        :return: Средняя длина слова.
        """
        words = self.tokenize_words()
        len_words = [len(word) for word in words]
        return sum(len_words) / len(words) if len(words) != 0 else 0
    
    def get_average_syllables_word(self) -> float:
        """
        Рассчитывает среднее количество слогов на слово в тексте.

        :return: Среднее количество слогов на слово.
        """
        words = self.tokenize_words()
        word_syllables_count = [self.get_syllables_count(word) for word in words]
        return sum(word_syllables_count) / len(words) if len(words) != 0 else 0
        
    def get_flesh_index(self) -> float:
        """
        Рассчитывает индекс удобочитаемости Флеша.

        Формула:
            Flesch Index = 206.835 - (1.015 * ASL) - (84.6 * ASW),
        где ASL — средняя длина предложения,
            ASW — среднее количество слогов на слово.

        :return: Значение индекса Флеша.
        """
        ASL = self.get_average_sentence_length()
        ASW = self.get_average_syllables_word()
        return 206.835 - (1.015 * ASL) - (84.6 * ASW)
    
    def get_bm25_scores(self, query) -> List[Tuple[str, float]]:
        """
        Рассчитывает BM25-оценки предложений по заданному запросу.

        :param query: Запрос, для которого вычисляются оценки.
        :return: Список кортежей (предложение, BM25-оценка).
        """
        sentences = self.tokenize_sentences()
        tokenized_docs = [word_tokenize(sentence.lower()) for sentence in sentences]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = word_tokenize(query.lower())
        return list(zip(sentences, bm25.get_scores(tokenized_query)))

    def analyze_lexical_lists(self) -> Dict[str, Any]:
        """
        Анализирует покрытие текста по лексическим словарям CEFR и по частотному словарю (freq_dict).

        Для каждого уровня (A1, A2, B1, B2, C1, C2) рассчитывается процент присутствующих слов,
        а также составляется список слов, отсутствующих в словаре.
        Для частотного словаря берутся ключи из freq_dict.

        :return: Словарь с информацией по каждому списку.
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
        # Анализ для частотного словаря freq_dict
        corpus_words = set(freq_dict.keys())
        intersection = text_unique.intersection(corpus_words)
        coverage = (len(intersection) / len(text_unique)) * 100 if text_unique else 0
        rare_words = sorted(list(text_unique - corpus_words))
        lexical_analysis["частотности"] = {
            "coverage": round(coverage),
            "rare_words": rare_words
        }
        return lexical_analysis

    def determine_actfl_level(self) -> str:
        """
        Определяет уровень текста по системе ACTFL с помощью AI (g4f).
        
        :return: Строка с определённым уровнем (например, "Advanced Mid", "Intermediate High" и т.д.).
        """
        try:
            prompt = f"""
            Определи уровень сложности этого русского текста по системе ACTFL. 
            Варианты уровней: "Advanced Mid", "Advanced Low", "Intermediate High", "Intermediate Mid", "Novice High", "Below Novice".
            Ответь только названием уровня, без дополнительных пояснений.
            
            Текст: {' '.join(self.words)}
            """
            
            response = g4f.ChatCompletion.create(
                model="gpt-4o",
                provider=Blackbox,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Очищаем и проверяем ответ
            level = response.strip()
            valid_levels = {
                "Advanced Mid", "Advanced Low", "Intermediate High",
                "Intermediate Mid", "Novice High", "Below Novice"
            }
            
            if level in valid_levels:
                return level
            return "Intermediate Mid"
            
        except Exception as e:
            print(f"Error in ACTFL determination: {e}")
            return "Intermediate Mid" 

    def actfl_to_cefr(actfl_level: str) -> str:
        """
        Переводит уровень сложности по системе ACTFL в соответствующий уровень по системе CEFR.
        
        :param actfl_level: Уровень ACTFL.
        :return: Соответствующий уровень CEFR
        """

        return ACTFL_TO_CEFR.get(actfl_level, None)

    def get_key_words(self) -> List[str]:
        """
        Выделяет ключевые слова на основе метрики TF/IDF.
        Для каждого слова TF рассчитывается по лемматизированному списку (self.normalized_words),
        а затем его оценка = (TF) / (частота из freq_dict + ε).
        Слова сортируются по убыванию оценки – чем выше оценка, тем слово характернее для данного текста.

        :return: Список ключевых слов, отсортированных по значимости.
        """
        tf = {}
        for word in self.normalized_words:
            tf[word] = tf.get(word, 0) + 1
        epsilon = 0.1  # корректирующий коэффициент для избежания деления на 0
        scores = {}
        for word, count in tf.items():
            corpus_freq = freq_dict.get(word, epsilon)
            scores[word] = count / corpus_freq
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words]

    def get_most_useful_words(self) -> List[str]:
        """
        Выделяет «самые полезные слова» – те, которых нет в базовых словарях (A1 ∪ A2),
        и сортирует их по значению TF/IDF (рассчитывается как (TF) / (частота из freq_dict + ε)).

        :return: Список полезных слов, отсортированный по убыванию значимости.
        """
        basic_vocab = a1.union(a2)
        tf = {}
        for word in self.normalized_words:
            tf[word] = tf.get(word, 0) + 1
        epsilon = 0.1
        scores = {}
        for word, count in tf.items():
            if word not in basic_vocab:
                corpus_freq = freq_dict.get(word, epsilon)
                scores[word] = count / corpus_freq
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words]

    def get_full_analysis_text(self) -> str:
        """
        Форматирует итоговый анализ текста в виде многострочной строки.

        :return: Отформатированная строка с детальным анализом текста.
        """
        actfl_level = self.determine_actfl_level()
        cefr_level = self.actfl_to_cefr(actfl_level)
        reading_times = self.calculate_reading_time()
        
        lines = [
            f"<b>Уровень текста в системе ACTFL</b> - {actfl_level}",
            f"<b>Уровень текста в системе CEFR</b> - {cefr_level}",
            f"<b>Знаков с пробелами</b> - {self.character_count}",
            f"<b>Предложений</b> - {self.sentence_count}",
            f"<b>Слов</b> - {self.word_count}",
            f"<b>Уникальных слов</b> - {self.get_unique_word_count()}",
            f"<b>Лексическое разнообразие</b> - {self.get_lexical_diversity()}\n",
            f"<b>Изучающее чтение</b>: {reading_times['study_time']}",
            f"<b>Просмотровое чтение</b>: {reading_times['skim_time']}",
            ]
        
        key_words = self.get_key_words()
        if key_words:
            lines.append("\n<b>Ключевые слова</b>:\n" + ", ".join(key_words))

        useful_words = self.get_most_useful_words()
        if useful_words:
            lines.append("\n<b>Самые полезные слова</b>:\n" + ", ".join(useful_words))

        lexical_lists = self.analyze_lexical_lists()
        for level, data in lexical_lists.items():
            lines.append(f"\n<b>Лексический словарь {level}</b> покрывает {data['coverage']}%")
            not_included = data.get("not_included", [])
            if not_included:
                lines.append(f"<b>Не входит в лексический словарь {level}</b>: " + ", ".join(not_included[:10]) + (
                    "..." if len(not_included) > 10 else ""))

        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    input_text = sys.stdin.read().strip()

    analyzer = TextAnalyzer(input_text)
    print(analyzer.get_full_analysis_text())
