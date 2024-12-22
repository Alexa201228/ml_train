from hamming_complex import RecurrentHammingNetwork


class AdaptiveHammingNetwork(RecurrentHammingNetwork):
    def update_template(self, input_vector, winner_index, eta=0.1):
        """
        Адаптивное обновление шаблона-победителя.
        :param input_vector: Входной вектор.
        :param winner_index: Индекс шаблона-победителя.
        :param eta: Коэффициент обучения.
        """
        self.templates[winner_index] = (
            (1 - eta) * self.templates[winner_index] + eta * input_vector
        )

# Пример использования
if __name__ == "__main__":
    # Шаблоны
    templates = [
        [1, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0],
    ]

    # Входной вектор
    input_vector = [1, 0, 1, 0, 1, 0]

    # Создание адаптивной сети
    network = AdaptiveHammingNetwork(templates, alpha=0.5, iterations=5)
    winner_index, _ = network.predict(input_vector)
    print(f"Победитель до обновления: T{winner_index + 1}")

    # Обновление шаблона
    network.update_template(input_vector, winner_index, eta=0.2)
    print("Обновленные шаблоны:")
    print(network.templates)
