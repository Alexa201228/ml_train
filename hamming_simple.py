import numpy as np

class HammingNetwork:
    def __init__(self, templates):
        """
        Инициализация сети с заданными шаблонами.
        :param templates: Список шаблонов (векторы эталонов).
        """
        self.templates = np.array(templates)
        self.num_templates = len(templates)
        self.num_features = len(templates[0])

    def hamming_distance(self, x, t):
        """
        Вычисление расстояния Хэмминга между входным вектором x и шаблоном t.
        :param x: Входной вектор.
        :param t: Шаблон.
        :return: Расстояние Хэмминга.
        """
        return np.sum(np.abs(x - t))

    def compare(self, input_vector):
        """
        Сравнение входного вектора с шаблонами и выбор наиболее похожего.
        :param input_vector: Входной вектор.
        :return: Индекс шаблона-победителя.
        """
        similarities = []
        for template in self.templates:
            # Количество совпадений = длина вектора - расстояние Хэмминга
            distance = self.hamming_distance(input_vector, template)
            similarity = self.num_features - distance
            similarities.append(similarity)
        
        # Нахождение индекса шаблона с максимальным значением похожести
        winner_index = np.argmax(similarities)
        return winner_index, similarities

# Пример использования
if __name__ == "__main__":
    # Шаблоны сети
    templates = [
        [1, 0, 1, 0, 1, 1],  # Шаблон T1
        [0, 1, 0, 1, 0, 1],  # Шаблон T2
        [1, 1, 1, 0, 1, 0],  # Шаблон T3
    ]

    # Входной вектор
    input_vector = [1, 0, 1, 0, 1, 1]

    # Инициализация и запуск сети
    network = HammingNetwork(templates)
    winner_index, similarities = network.compare(input_vector)

    print(f"Входной вектор: {input_vector}")
    print(f"Сходства с шаблонами: {similarities}")
    print(f"Победивший шаблон: T{winner_index + 1}")
