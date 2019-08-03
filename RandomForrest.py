from numpy import int32, float32

from cv2.ml import RTrees_create, ROW_SAMPLE
from cv2 import TermCriteria_MAX_ITER, TermCriteria_EPS

from pandas import read_csv

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(data_file):
    data = read_csv(data_file, dtype=None, delimiter=',')
    samples = data.iloc[:, 2:31].values
    response = data.iloc[:, 1].values
    return samples, response


def set_params(classifier, active_var, min_sample):
    classifier.setActiveVarCount(active_var)
    classifier.setMinSampleCount(min_sample)
    classifier.setTermCriteria((TermCriteria_MAX_ITER | TermCriteria_EPS, 500, 0.001))

def print_results(response, results, performance):
    results = ['B' if y == 0 else 'M' for y in results]
    correct_pred = int(round(performance*len(response)))
    incorrect_pred = len(response)-correct_pred
    for x, y in zip(response, results):
        print(" Real value:", x, "  Predicted value:", y)
    print("\n", "Correct predictions:", correct_pred, "  Incorrect predictions:", incorrect_pred)
    print("\n", "Algorithm accuracy is:", round(performance*100, 2), "%")


def main():
    data_file = open('/home/wolfie/Desktop/Završni/RFA-py/breast_cancer_dataset.csv')
    samples, response = load_data(data_file)
    le = LabelEncoder()

    samples = float32(samples)
    response = int32(le.fit_transform(response))

    samples_train, samples_test, response_train, response_test = train_test_split(samples, response, test_size=0.2)

    rf_classifier = RTrees_create()

    set_params(rf_classifier, 0, 2)

    rf_classifier.train(samples_train, ROW_SAMPLE, response_train)

    var, results = rf_classifier.predict(samples_test)

    performance = accuracy_score(response_test, results)

    response_test = le.inverse_transform(response_test)

    print_results(response_test, results, performance)


if __name__ == "__main__":
    main()
