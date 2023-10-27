# Author: Naufal Suryanto (https://github.com/naufalso)

import tensorflow as tf
import numpy as np


class TripletMetrics:
    def __init__(
        self,
        batch_size: int,
        min_thres: float = 0.3,
        max_thres: float = 0.81,
        interval: float = 0.1,
    ):
        """
        Initializes the TripletMetrics class with the given parameters.

        Args:
            batch_size (int): The batch size used for training.
            min_thres (float): The minimum threshold value for accuracy metrics.
            max_thres (float): The maximum threshold value for accuracy metrics.
            interval (float): The interval between threshold values for accuracy metrics.
        """
        self.batch_size = batch_size
        self.min_thres = min_thres
        self.max_thres = max_thres
        self.interval = interval

    def get_distance_metrics(self):
        """
        Returns a list of distance metrics for positive and negative pairs.

        Returns:
            List: A list of distance metrics for positive and negative pairs.
        """
        return [
            DistanceOfPairs(self.batch_size, mode="max", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="min", pairs="neg"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="neg"),
        ]

    def get_accuracy_metrics(self):
        """
        Returns a list of accuracy metrics for positive and negative pairs.

        Returns:
            List: A list of accuracy metrics for positive and negative pairs.
        """
        positive_acc = [
            AccOfPairs(self.batch_size, threshold=thres, pairs="pos")
            for thres in np.arange(self.min_thres, self.max_thres, self.interval)
        ]
        negative_acc = [
            AccOfPairs(self.batch_size, threshold=thres, pairs="neg")
            for thres in np.arange(self.min_thres, self.max_thres, self.interval)
        ]
        return positive_acc + negative_acc


class DistanceOfPairs(tf.keras.metrics.Metric):
    def __init__(self, batch_size, mode="max", pairs="pos", **kwargs):
        """
        Initializes the DistanceOfPairs class with the given parameters.

        Args:
            batch_size (int): The batch size used for training.
            mode (str): The mode of aggregation for the distance metric. Can be "max", "min", or "avg".
            pairs (str): The type of pairs to compute the distance metric for. Can be "pos" or "neg".
        """
        name = f"dist/{pairs}/{mode}"
        super(DistanceOfPairs, self).__init__(name=name, **kwargs)

        assert mode in ["max", "min", "avg"]
        assert pairs in ["pos", "neg"]

        aggregate_funcs = {
            "max": tf.math.reduce_max,
            "min": tf.math.reduce_min,
            "avg": tf.math.reduce_mean,
        }

        self.batch_size = batch_size
        self.aggregate_func = aggregate_funcs[mode]
        self.pair_distance_func = (
            self._positive_distance if pairs == "pos" else self._negative_distance
        )

        self.mean_distances = tf.keras.metrics.Mean(f"mean_{name}")

    def _positive_distance(self, grouped_embedding):
        """
        Computes the distance metric for positive pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings for each pair.

        Returns:
            tf.Tensor: A tensor of shape [BATCH_SIZE] containing the distance metric for each pair.
        """
        positive_distances = tf.norm(
            grouped_embedding[:, 0, :] - grouped_embedding[:, 1, :], axis=1
        )

        aggregated_distance = self.aggregate_func(positive_distances)
        return aggregated_distance

    def _negative_distance(self, grouped_embedding):
        """
        Computes the distance metric for negative pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings for each pair.

        Returns:
            tf.Tensor: A tensor of shape [2, BATCH_SIZE-1] containing the distance metric for each pair.
        """
        negative_distance_ori = tf.norm(
            grouped_embedding[:-1, 0, :] - grouped_embedding[1:, 0, :], axis=1
        )
        negative_distance_aug = tf.norm(
            grouped_embedding[:-1, 1, :] - grouped_embedding[1:, 1, :], axis=1
        )
        negative_distance = tf.stack([negative_distance_ori, negative_distance_aug])

        aggregated_distance = self.aggregate_func(negative_distance)
        return aggregated_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric with new data.

        Args:
            y_true (tf.Tensor): A tensor of shape [BATCH_SIZE] containing the true labels for each embedding.
            y_pred (tf.Tensor): A tensor of shape [BATCH_SIZE, EMBEDDING_SIZE] containing the predicted embeddings.
            sample_weight (tf.Tensor): Optional weighting for each sample.
        """
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        grouped_embedding = tf.stack(tf.split(embeddings, 2), axis=1)

        aggregated_distance = self.pair_distance_func(grouped_embedding)
        self.mean_distances.update_state(aggregated_distance)

    def result(self):
        """
        Computes and returns the result of the metric.

        Returns:
            tf.Tensor: The result of the metric.
        """
        return self.mean_distances.result()

    def reset_state(self):
        """
        Resets the state of the metric.
        """
        self.mean_distances.reset_state()

    def get_distance_metrics(self):
        """
        Returns a list of distance metrics for positive and negative pairs.

        Returns:
            List: A list of distance metrics for positive and negative pairs.
        """
        return [
            DistanceOfPairs(self.batch_size, mode="max", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="min", pairs="neg"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="pos"),
            DistanceOfPairs(self.batch_size, mode="avg", pairs="neg"),
        ]

    def get_accuracy_metrics(self):
        """
        Returns a list of accuracy metrics for positive and negative pairs.

        Returns:
            List: A list of accuracy metrics for positive and negative pairs.
        """
        positive_acc = [
            AccOfPairs(self.batch_size, threshold=thres, pairs="pos")
            for thres in np.arange(self.min_thres, self.max_thres, self.interval)
        ]
        negative_acc = [
            AccOfPairs(self.batch_size, threshold=thres, pairs="neg")
            for thres in np.arange(self.min_thres, self.max_thres, self.interval)
        ]
        return positive_acc + negative_acc


class DistanceOfPairs(tf.keras.metrics.Metric):
    def __init__(self, batch_size, mode="max", pairs="pos", **kwargs):
        """
        Initializes the DistanceOfPairs metric.

        Args:
            batch_size (int): The batch size.
            mode (str): The mode of the metric. Can be "max", "min", or "avg".
            pairs (str): The type of pairs to compute the distance for. Can be "pos" for positive pairs or "neg" for negative pairs.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
            None
        """
        name = f"dist/{pairs}/{mode}"
        super(DistanceOfPairs, self).__init__(name=name, **kwargs)

        assert mode in ["max", "min", "avg"]
        assert pairs in ["pos", "neg"]

        aggregate_funcs = {
            "max": tf.math.reduce_max,
            "min": tf.math.reduce_min,
            "avg": tf.math.reduce_mean,
        }

        self.batch_size = batch_size
        self.aggregate_func = aggregate_funcs[mode]
        self.pair_distance_func = (
            self._positive_distance if pairs == "pos" else self._negative_distance
        )

        self.mean_distances = tf.keras.metrics.Mean(f"mean_{name}")

    def _positive_distance(self, grouped_embedding):
        """
        Computes the distance between positive pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings of the pairs.

        Returns:
            aggregated_distance (tf.Tensor): A scalar tensor containing the aggregated distance between the positive pairs.
        """
        positive_distances = tf.norm(
            grouped_embedding[:, 0, :] - grouped_embedding[:, 1, :], axis=1
        )

        aggregated_distance = self.aggregate_func(positive_distances)
        return aggregated_distance

    def _negative_distance(self, grouped_embedding):
        """
        Computes the distance between negative pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings of the pairs.

        Returns:
            aggregated_distance (tf.Tensor): A scalar tensor containing the aggregated distance between the negative pairs.
        """
        negative_distance_ori = tf.norm(
            grouped_embedding[:-1, 0, :] - grouped_embedding[1:, 0, :], axis=1
        )
        negative_distance_aug = tf.norm(
            grouped_embedding[:-1, 1, :] - grouped_embedding[1:, 1, :], axis=1
        )
        negative_distance = tf.stack([negative_distance_ori, negative_distance_aug])

        aggregated_distance = self.aggregate_func(negative_distance)
        return aggregated_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates the distance between pairs.

        Args:
            y_true (tf.Tensor): A tensor of shape [BATCH_SIZE, 2] containing the labels of the pairs.
            y_pred (tf.Tensor): A tensor of shape [BATCH_SIZE, EMBEDDING_SIZE] containing the embeddings of the pairs.
            sample_weight (tf.Tensor): Optional weighting of each example. Defaults to None.

        Returns:
            None
        """
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        # Group the same labels
        grouped_embedding = tf.stack(tf.split(embeddings, 2), axis=1)

        aggregated_distance = self.pair_distance_func(grouped_embedding)
        self.mean_distances.update_state(aggregated_distance)

    def result(self):
        """
        Computes and returns the mean distance between pairs.

        Args:
            None

        Returns:
            mean_distances (tf.Tensor): A scalar tensor containing the mean distance between pairs.
        """
        return self.mean_distances.result()

    def reset_state(self):
        """
        Resets the metric's state.

        Args:
            None

        Returns:
            None
        """
        self.mean_distances.reset_state()


class AccOfPairs(tf.keras.metrics.Metric):
    def __init__(self, batch_size, threshold=0.65, pairs="pos", **kwargs):
        """
        Initializes the AccOfPairs class.

        Args:
            batch_size (int): The batch size.
            threshold (float): The threshold value for the accuracy.
            pairs (str): The type of pairs to compute the accuracy for. Can be "pos" or "neg".
            **kwargs: Additional arguments to pass to the parent class.

        Returns:
            None
        """
        name = f"acc/{pairs}/{threshold:.2f}"
        super(AccOfPairs, self).__init__(name=name, **kwargs)

        assert pairs in ["pos", "neg"]

        self.batch_size = batch_size
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        self.pair_acc_func = (
            self._positive_acc if pairs == "pos" else self._negative_acc
        )

        self.mean_acc = tf.keras.metrics.Mean(f"mean_{name}")

    def _positive_acc(self, grouped_embedding):
        """
        Computes the accuracy for positive pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings of the pairs.

        Returns:
            acc (tf.Tensor): A scalar tensor containing the accuracy for positive pairs.
        """
        # Positive pair distance shape: [BATCH_SIZE]
        positive_distances = tf.norm(
            grouped_embedding[:, 0, :] - grouped_embedding[:, 1, :], axis=1
        )
        correct_preds = tf.where(positive_distances <= self.threshold, 1.0, 0.0)

        return tf.reduce_mean(correct_preds)

    def _negative_acc(self, grouped_embedding):
        """
        Computes the accuracy for negative pairs.

        Args:
            grouped_embedding (tf.Tensor): A tensor of shape [BATCH_SIZE, 2, EMBEDDING_SIZE] containing the embeddings of the pairs.

        Returns:
            acc (tf.Tensor): A scalar tensor containing the accuracy for negative pairs.
        """
        # Negative pair distance shape: [2, BATCH_SIZE-1]
        negative_distance_ori = tf.norm(
            grouped_embedding[:-1, 0, :] - grouped_embedding[1:, 0, :], axis=1
        )
        negative_distance_aug = tf.norm(
            grouped_embedding[:-1, 1, :] - grouped_embedding[1:, 1, :], axis=1
        )
        negative_distance = tf.stack([negative_distance_ori, negative_distance_aug])

        correct_preds = tf.where(negative_distance > self.threshold, 1.0, 0.0)

        return tf.reduce_mean(correct_preds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates the accuracy for pairs.

        Args:
            y_true (tf.Tensor): A tensor of shape [BATCH_SIZE, 2] containing the labels of the pairs.
            y_pred (tf.Tensor): A tensor of shape [BATCH_SIZE, EMBEDDING_SIZE] containing the embeddings of the pairs.
            sample_weight (tf.Tensor): Optional weighting of each example. Defaults to None.

        Returns:
            None
        """
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        # Group the same labels
        grouped_embedding = tf.stack(tf.split(embeddings, 2), axis=1)

        aggregated_acc = self.pair_acc_func(grouped_embedding)
        self.mean_acc.update_state(aggregated_acc)

    def result(self):
        """
        Computes and returns the mean accuracy for pairs.

        Args:
            None

        Returns:
            mean_acc (tf.Tensor): A scalar tensor containing the mean accuracy for pairs.
        """
        return self.mean_acc.result()

    def reset_state(self):
        """
        Resets the metric's state.

        Args:
            None

        Returns:
            None
        """
        self.mean_acc.reset_state()
