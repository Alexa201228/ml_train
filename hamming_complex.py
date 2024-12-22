import numpy as np

class RecurrentHammingNetwork:
    def __init__(self, templates, alpha=0.5, iterations=5):
        """
        Инициализация сети Хэмминга с обратными связями.
        :param templates: Список шаблонов (эталонов).
        :param alpha: Коэффициент затухания для обратных связей.
        :param iterations: Количество итераций для обновления нейронов.
        """
        self.templates = np.array(templates)
        self.num_templates = len(templates)
        self.num_features = len(templates[0])
        self.alpha = alpha
        self.iterations = iterations

    def hamming_distance(self, x, t):
        """Расстояние Хэмминга между вектором x и шаблоном t."""
        return np.sum(np.abs(x - t))

    def forward_pass(self, input_vector):
        """
        Прямой проход и итеративное обновление с обратными связями.
        :param input_vector: Входной вектор.
        :return: Значения нейронов на выходе.
        """
        # Начальные значения для нейронов второго слоя
        y = np.zeros(self.num_templates)

        # Итеративное обновление с обратными связями
        for iteration in range(self.iterations):
            print(f"\nИтерация {iteration + 1}:")
            for i, template in enumerate(self.templates):
                # Сумма текущей похожести и обратной связи
                distance = self.hamming_distance(input_vector, template)
                y[i] = (self.num_features - distance) + self.alpha * y[i]
                print(f"Шаблон T{i + 1}: значение нейрона = {y[i]:.2f}")
        
        return y

    def predict(self, input_vector):
        """
        Получение результата сети.
        :param input_vector: Входной вектор.
        :return: Индекс шаблона-победителя.
        """
        output_values = self.forward_pass(input_vector)
        winner_index = np.argmax(output_values)
        return winner_index, output_values

# Пример использования
if __name__ == "__main__":
    # Шаблоны сети
    templates = [
        [1, 0, 1, 0, 1, 1],  # Шаблон T1
        [0, 1, 0, 1, 0, 1],  # Шаблон T2
        [1, 1, 1, 0, 1, 0],  # Шаблон T3
    ]

    # Входной вектор
    input_vector = [1, 0, 1, 0, 1, 0]

    # Инициализация и запуск сети
    network = RecurrentHammingNetwork(templates, alpha=0.5, iterations=5)
    winner_index, output_values = network.predict(input_vector)

    print("\nФинальные значения нейронов второго слоя:", output_values)
    print(f"Победивший шаблон: T{winner_index + 1}")


