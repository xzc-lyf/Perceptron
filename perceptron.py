import math

'''
:param
    file_path: The dataset file path.
:returns
    features: A list used to store the coordinates in the dataset.
        ['-8.459748692855033', '5.621530829461882'],
        ['4.237208820754372', '-6.462792989330957'],
        ['11.644215057276138', '9.451504823451543'],
        ['12.66037789730385', '-3.6062685957325904'], ...
    labels: A list stores the label (1 or -1) for each data point.
        [-1, 1, -1, 1, ...]
    dimension: Number of features (2, 4, 8).
    radius: radius (16, 24, 12).
'''
def load_dataset(file_path):
    features, labels = [], []
    with open(file_path, 'r') as dataset:
        lines = dataset.readlines()
        metadata = lines[0].strip().split(',')
        dimension, radius = int(metadata[0]), int(metadata[-1])
        for line in lines[1:]:
            values = line.strip().split(',')
            features.append(values[:dimension])
            labels.append(int(values[-1]))
    return features, labels, dimension, radius

'''
:param
    vec1: The vector1 to be computed.
    vec2: The vector2 to be computed.
:returns
    res: The result of dot product.
'''
def compute_dot_product(vec1, vec2):
    return sum(float(a) * float(b) for a, b in zip(vec1, vec2))

"""
:param
    radius: The points of S fall in a ball that centers at the origin and has radius R.
    gamma: Denote by γ the largest margin of all the separation planes.
:returns
    res: Maximum number of iterations.
"""
def calculate_max_iterations(radius, gamma):
    return math.ceil(12 * (radius ** 2) / (gamma ** 2))

"""
:param
    weights: The weight vector to be adjusted.
    features: A list used to store the coordinates in the dataset.
    labels: A list stores the label (1 or -1) for each data point.
:returns
    res: Maximum number of iterations.
"""
def update_weights(weights, point, label):
    return [w + label * float(x) for w, x in zip(weights, point)]

"""
:param
    features: A list used to store the coordinates in the dataset.
    labels: A list stores the label (1 or -1) for each data point.
    weights: The weight vector to be adjusted.
    gamma_guess: An arbitrary value (Set to radius in this case).
:returns
    res1: Index.
    res2: Distance between the violation point and the plane.
"""
def check_violation(features, labels, weights, gamma_guess):
    len_w = math.sqrt(sum(w ** 2 for w in weights)) if any(weights) else 0
    for index, (point, label) in enumerate(zip(features, labels)):
        dot_val = compute_dot_product(weights, point)
        curr_margin = math.fabs(dot_val / len_w) if len_w else 0
        if (curr_margin < gamma_guess / 2) or (label == 1 and dot_val <= 0) or (label == -1 and dot_val >= 0):
            return index, curr_margin
    return -1, curr_margin


"""
:param
    features: A list used to store the coordinates in the dataset.
    labels: A list stores the label (1 or -1) for each data point.
    max_iterations: Maximum number of iterations.
    gamma_guess: gamma_guess.
    dimension: Number of features.
:returns
    res1: The weight vector.
    res2: The flag of training complete or not.
"""
def train_model(features, labels, max_iterations, gamma_guess, weights):

    for iteration in range(max_iterations):
        violation_index, margin = check_violation(features, labels, weights, gamma_guess)

        print(f"\nIteration {iteration + 1}")
        print("Current Weights:", weights)

        if violation_index == -1:
            print("Training complete - no violations.")
            print(f"Final Weights after {iteration} iterations:", weights)
            return weights, True

        print("Margin of violating sample:", margin)
        print("Violating sample:", features[violation_index])
        print("Label of violating sample:", labels[violation_index])

        weights = update_weights(weights, features[violation_index], labels[violation_index])

        print("Weights after update:", weights)
        print('-' * 40)

    print("Training stopped - maximum iterations reached.")
    return weights, False


"""
:param
    file_list: The dataset file path list.
"""
def train_margin_perceptron(file_list):
    final_gammas = []
    final_weights = []

    for file in file_list:
        features, labels, dimension, radius = load_dataset(file)
        gamma_guess = radius
        max_iterations = calculate_max_iterations(radius, gamma_guess)

        weights = [0] * dimension
        while True:
            weights, training_complete = train_model(features, labels, max_iterations, gamma_guess, weights)
            if training_complete or gamma_guess <= 1e-8:
                if gamma_guess <= 1e-8:
                    print("Converged to approximate requirement; stopping margin perceptron.")
                break

            gamma_guess /= 2
            max_iterations = calculate_max_iterations(radius, gamma_guess)

        final_gammas.append(gamma_guess)
        final_weights.append(weights)
        print(f"Final gamma guess: {gamma_guess}")
        print("Final weights:", weights)

    print('-' * 40)
    for i, (gamma, weight) in enumerate(zip(final_gammas, final_weights)):
        print(f"\nFile {i+1}:")
        print(f"Final gamma guess: {gamma}")
        print(f"Final weights: {weight}")

train_margin_perceptron(["./Dataset1.txt", "./Dataset2.txt","./Dataset3.txt"])
