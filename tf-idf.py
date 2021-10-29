from math import log
from collections import defaultdict


class CountVectorizer():
    def get_feature_names(self) -> list:
        """возвращаем список всех слов из текста"""
        list_of_names = []
        check = set()
        for row in self.corpus:
            for word in row.lower().split():
                word = word.rstrip('.,!?')
                if word not in check:
                    list_of_names.append(word)
                    check.add(word)
        return list_of_names

    def fit_transform(self, corpus) -> list:
        """возвращаем терм-документную матрицу"""
        matrix = []
        self.corpus = corpus
        feature_names = self.get_feature_names()
        pos_name_matrix = tuple(enumerate(feature_names))
        for row in corpus:
            counter = len(feature_names) * [0]

            check = defaultdict(int)
            for word in row.lower().split():
                check[word.rstrip('.,!?')] += 1

            for index, name in pos_name_matrix:
                counter[index] = check[name]
            matrix.append(counter)
        return matrix


class TfidfVectorizer(CountVectorizer):
    def fit_transform(self, corpus):
        """переопредеяем этот метод на основе родителького класса"""
        count_matrix = super().fit_transform(corpus)
        transformer = TfidfTransformer()
        return transformer.fit_transform(count_matrix)

class TfidfTransformer():
    def fit_transform(self, count_matrix):
        """возвращаем tf-idf матрицу"""
        res = []
        idf_matrix = idf_transform(count_matrix)
        tf_matrix = tf_transform(count_matrix)
        for document in tf_matrix:
            matrix = []
            for i, j in zip(document, idf_matrix):
                matrix.append(round(i * j, 3))
            res.append(matrix)
        return res


def tf_transform(count_matrix):
    """возвращаем матрицу term-frequency"""
    tf = []
    for document in count_matrix:
        length = sum(document)
        res = [round(word / length, 3) for word in document]
        tf.append(res)
    return tf


def idf_transform(count_matrix):
    """возвращаем список inverse document-frequency"""
    word_doc_freq = []
    length = len(count_matrix)
    for index in range(len(count_matrix[0])):
        counter = 0
        for doc in count_matrix:
            if doc[index]:
                counter += 1
        word_doc_freq.append(counter)
    res = [round(log((length + 1) / (word_freq + 1)), 3) + 1 for word_freq in word_doc_freq]
    return res


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
