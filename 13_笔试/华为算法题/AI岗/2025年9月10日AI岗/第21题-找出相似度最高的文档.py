import math
import sys
from collections import Counter, defaultdict

def solve():
    input = sys.stdin.buffer.readline
    document_count = int(input())
    documents = [
        Counter(input().decode().strip().lower().split())
        for _ in range(document_count)
    ]
    window_size = int(input())
    query_count = int(input())

    queries_by_end = defaultdict(list)
    for query_id in range(query_count):
        parts = input().decode().strip().lower().split()
        query_time = int(parts[0])
        query = Counter(parts[1:])
        window_end = min(query_time, document_count - 1)
        queries_by_end[window_end].append((query_id, query))

    answers = [-1] * query_count
    threshold = 0.6
    epsilon = 1e-12

    for window_end, queries in queries_by_end.items():
        window_start = window_end - window_size + 1
        window = documents[window_start : window_end + 1]

        document_frequency = Counter()
        for document in window:
            document_frequency.update(document.keys())

        def idf(word):
            return (
                math.log(
                    (window_size + 1)
                    / (document_frequency.get(word, 0) + 1)
                )
                + 1.0
            )

        weighted_documents = []
        inverted_index = defaultdict(list)
        for local_id, document in enumerate(window):
            vector = {}
            norm_square = 0.0
            for word, frequency in document.items():
                value = frequency * idf(word)
                vector[word] = value
                norm_square += value * value
                inverted_index[word].append((local_id, value))
            weighted_documents.append(
                (math.sqrt(norm_square), (local_id + 1) / window_size)
            )

        for query_id, query in queries:
            query_values = {}
            query_norm_square = 0.0
            for word, frequency in query.items():
                value = frequency * idf(word)
                query_values[word] = value
                query_norm_square += value * value

            if query_norm_square == 0:
                continue

            dot_products = [0.0] * window_size
            for word, query_value in query_values.items():
                for local_id, document_value in inverted_index.get(word, ()):
                    dot_products[local_id] += query_value * document_value

            query_norm = math.sqrt(query_norm_square)
            best_score = -1.0
            best_document = -1

            for local_id, dot_product in enumerate(dot_products):
                document_norm, time_weight = weighted_documents[local_id]
                if document_norm == 0:
                    continue
                cosine = dot_product / (query_norm * document_norm)
                if cosine + epsilon < threshold:
                    continue

                score = cosine * time_weight
                if score > best_score + epsilon:
                    best_score = score
                    best_document = window_start + local_id

            answers[query_id] = best_document

    print(*answers)

if __name__ == "__main__":
    solve()
