import numpy as np

def calc_binary_precision(predictions, labels):

    tps = np.sum(predictions * labels)
    fps = np.sum(predictions) - tps

    return tps / (tps + fps)


def calc_binary_sensitivity(predictions, labels):

    tps = np.sum(predictions * labels)
    tns = np.sum(labels[np.nonzero(predictions == 0)] == 0)
    fns = np.sum(predictions == 0) - tns

    return tps / (tps + fns)

def calc_binary_specificity(predictions, labels):

    tns = np.sum(labels[np.nonzero(predictions == 0)] == 0)
    tps = np.sum(predictions * labels)
    fps = np.sum(predictions == 1) - tps

    return tns / (tns + fps)

def calc_binary_f1_score(predictions, labels):

    precision = calc_binary_precision(predictions, labels)
    recall = calc_binary_sensitivity(predictions, labels)

    return (precision * recall) / (precision + recall)

def calc_binary_percent_accuracy(predictions, labels):

    return np.sum(predictions == labels) / len(predictions) * 100

def print_metrics(predictions, labels) -> str:
    output_str  = "\n".join((
                f"{'-' * 30}",
                f"Percent Accuracy: {calc_binary_percent_accuracy(predictions, labels)}",
                f"Precision: {calc_binary_precision(predictions, labels)}",
                f"Sensitivity: {calc_binary_sensitivity(predictions, labels)}",
                f"Specificity: {calc_binary_specificity(predictions, labels)}",
                f"F1 Score: {calc_binary_f1_score(predictions, labels)}",
                f"{'-' * 30}",
        ))

    return output_str


if __name__ == "__main__":
    print(calc_binary_precision(np.array([1,0,1,1,0,0,1,1,0,1]), np.array([1,0,0,1,0,1,0,1,1,0])))
    print(calc_binary_percent_accuracy(np.array([1,0,1,1,0,0,1,1,0,1]), np.array([1,0,0,1,0,1,0,1,1,0])))


