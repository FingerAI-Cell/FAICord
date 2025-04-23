from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNCluster:
    def relabel_by_knn(self, embeddings, labels, k=5):
        """
        embeddings: numpy array [N, D]
        labels: list or np.array of str, 화자 레이블
        k: int, 주변 이웃 수
        return: new_labels (리스트)
        """
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        new_labels = []

        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')    # +1은 자기 자신 포함
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings)

        for i in range(len(embeddings)):
            neighbor_idx = indices[i][1:]    # 자기 자신 제외
            neighbor_labels = labels[neighbor_idx]
            majority_label = np.bincount([ord(label) for label in neighbor_labels]).argmax()
            new_labels.append(chr(majority_label))
        return new_labels