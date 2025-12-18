import torch
import numpy as np
import json


def load_rnn_params(filepath):
    with open(filepath, 'r') as f:
         data = json.load(f)
    return torch.tensor(data, dtype=torch.float32)


W_h = load_rnn_params("W_h.weight.json")
Wh_bias = load_rnn_params("W_h.bias.json").unsqueeze(1)
U_h = load_rnn_params("U_h.weight.json")
W_y = load_rnn_params("W_y.weight.json")
Embedding = load_rnn_params("embedding.weight.json")

with open("vocab.json", 'r') as f:
    VOCAB = json.load(f)

# Знаходимо індекси спеціальних токенів
BOS_INDEX = VOCAB.index('[')
EOS_INDEX = VOCAB.index(']')

# Розміри
embedding_dim = 96
hidden_dim = 160
vocab_size = 132


@torch.no_grad()
def greedy_decode_rnn(W_h, Wh_bias, U_h, W_y, Embedding, VOCAB, BOS_INDEX, EOS_INDEX, max_len=100):
    # Ініціалізація
    h_t = torch.zeros(hidden_dim, 1, dtype=torch.float32)

    # Початковий токен
    current_token_index = BOS_INDEX

    decoded_message_indices = [BOS_INDEX]

    # Цикл декодування
    for _ in range(max_len - 1):  # -1, тому що перший токен вже є

        # 1. Вхідний вектор (x_t)
        # Розмірність: (embedding_dim) -> [96]
        x_t = Embedding[current_token_index]
        # Додаємо вимір для матричного множення: [96] -> [96, 1]
        x_t = x_t.unsqueeze(1)

        # 2. Обчислення нового прихованого стану (h_t)
        # Використовуємо оператор @ для матричного множення (MatMul)

        # W_h @ x_t: [160, 96] @ [96, 1] -> [160, 1]
        term1 = W_h @ x_t

        # U_h @ h_{t-1}: [160, 160] @ [160, 1] -> [160, 1]
        term2 = U_h @ h_t

        # Новий прихований стан (h_t)
        h_t = torch.tanh(term1 + Wh_bias + term2)

        # 3. Обчислення логітів виходу (y_t)
        # y_t: [vocab_size, hidden_dim] @ [hidden_dim, 1] -> [vocab_size, 1]
        y_t = W_y @ h_t

        # 4. Жадібний декодинг (argmax)
        # Знаходимо індекс токена з найбільшим логітом
        current_token_index = torch.argmax(y_t).item()

        # 5. Зупинка
        if current_token_index == EOS_INDEX:
            decoded_message_indices.append(current_token_index)
            break

        decoded_message_indices.append(current_token_index)

    # 6. Декодування повідомлення
    decoded_message = "".join([VOCAB[idx] for idx in decoded_message_indices])

    return decoded_message


decoded_message = greedy_decode_rnn(W_h, Wh_bias, U_h, W_y, Embedding, VOCAB, BOS_INDEX, EOS_INDEX)
print(decoded_message)
