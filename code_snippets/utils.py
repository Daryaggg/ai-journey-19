from typing import Any

import matplotlib.pyplot as plt
from pymorphy2 import MorphAnalyzer


class MyMorph(MorphAnalyzer):
    def get_tag(self, word: str):
        return self.parse(word)[0].tag


def plot_learning_metrics(history) -> None:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    for ax, metric in zip([ax1, ax2], ['Accuracy', 'Loss']):
        ax.set_title(f'Model {metric}')
        ax.plot(history.history[metric.lower()], label='train', color='#326ba8', linewidth=2)
        ax.plot(
            history.history[f'val_{metric.lower()}'],
            label='validation',
            color='#32a85f',
            linewidth=2,
        )
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Epoch number')
        ax.legend()

    plt.show()


def predict_binary(model: Any, X: Any, threshold: int) -> List[int]:
    return [1 if _ > threshold else 0 for _ in model.predict(X)]
