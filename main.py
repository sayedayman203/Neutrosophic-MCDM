import functools
import operator
import numpy as np

from bwm import calc_weight
from neutrosophic_number import NeutrosophicNumber

scale = {
    "MO": NeutrosophicNumber(0.10, 0.25, 0.3, 0.1, 0.3, 0.1),
    "LO": NeutrosophicNumber(0.2, 0.3, 0.5, 0.6, 0.1, 0.2),
    "PO": NeutrosophicNumber(0.45, 0.3, 0.5, 0.6, 0.1, 0.2),
    "EO": NeutrosophicNumber(0.5, 0.5, 0.5, 0.9, 0.1, 0.1),
    "SO": NeutrosophicNumber(0.7, 0.75, 0.8, 0.9, 0.2, 0.2),
    "VSO": NeutrosophicNumber(0.85, 0.8, 0.95, 0.8, 0.1, 0.2),
    "AO": NeutrosophicNumber(0.95, 0.9, 0.95, 0.9, 0.1, 0.1),
}

criterias = [
    "Chills",
    "Nasal congestion",
    "Headache",
    "Cough",
    "Sore throat",
    "Sputum production",
    "Fatigue",
    "Shortness of breath",
    "Fever",
]

alternates = ["H1N1", "COVID-19", "H5N1", "Hanta Virus", "SARS"]


# BWM
compared2best = dict(
    {
        "Chills": 5,
        "Nasal congestion": 7,
        "Headache": 4,
        "Cough": 1,
        "Sore throat": 3,
        "Sputum production": 8,
        "Fatigue": 6,
        "Shortness of breath": 9,
        "Fever": 2,
    }
)
compared2worst = dict(
    {
        "Chills": 4,
        "Nasal congestion": 3,
        "Headache": 6,
        "Cough": 9,
        "Sore throat": 7,
        "Sputum production": 2,
        "Fatigue": 5,
        "Shortness of breath": 1,
        "Fever": 8,
    }
)

best_criteria = "Cough"
worst_criteria = "Shortness of breath"


weights = calc_weight(compared2best, compared2worst, best_criteria, worst_criteria)
weights_in_criterias_order = [weights[criteria] for criteria in criterias]
# print(weights_in_criterias_order)

# TOPSIS
decisions = np.array(
    [
        [
            [
                scale.get("EO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("AO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("VSO"),
            ],
            [
                scale.get("EO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("AO"),
                scale.get("VSO"),
                scale.get("PO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("AO"),
            ],
            [
                scale.get("SO"),
                scale.get("LO"),
                scale.get("EO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
        ],
        [
            [
                scale.get("PO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("AO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("AO"),
            ],
            [
                scale.get("SO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("VSO"),
            ],
            [
                scale.get("SO"),
                scale.get("MO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
            [
                scale.get("SO"),
                scale.get("LO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("PO"),
                scale.get("SO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("VSO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
        ],
        [
            [
                scale.get("SO"),
                scale.get("EO"),
                scale.get("VSO"),
                scale.get("AO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("AO"),
            ],
            [
                scale.get("EO"),
                scale.get("SO"),
                scale.get("SO"),
                scale.get("AO"),
                scale.get("VSO"),
                scale.get("PO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("AO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("PO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("MO"),
                scale.get("SO"),
                scale.get("VSO"),
                scale.get("VSO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("VSO"),
                scale.get("VSO"),
                scale.get("EO"),
                scale.get("SO"),
                scale.get("EO"),
                scale.get("VSO"),
                scale.get("AO"),
            ],
            [
                scale.get("VSO"),
                scale.get("LO"),
                scale.get("AO"),
                scale.get("SO"),
                scale.get("LO"),
                scale.get("LO"),
                scale.get("AO"),
                scale.get("SO"),
                scale.get("AO"),
            ],
        ],
    ]
)


def aggregate_decisions(decisions):
    num_decisions = len(decisions)
    num_alternatives = len(decisions[0])
    num_criteria = len(decisions[0][0])

    # Initialize the aggregated_decisions list with Empty Values
    aggregated_decisions = np.empty([num_alternatives, num_criteria], dtype=object)

    for alt_idx in range(num_alternatives):
        for crit_idx in range(num_criteria):
            aggregated_add_val = functools.reduce(
                operator.add, decisions[:, alt_idx, crit_idx]
            )

            aggregated_add_val.low = aggregated_add_val.low / num_decisions
            aggregated_add_val.mid = aggregated_add_val.mid / num_decisions
            aggregated_add_val.high = aggregated_add_val.high / num_decisions

            aggregated_decisions[alt_idx, crit_idx] = aggregated_add_val

    return aggregated_decisions


aggregated_results = aggregate_decisions(decisions)
# print(aggregated_results)

crisp_results = [
    [criteria.de_nutrosophication() for criteria in alternative]
    for alternative in aggregated_results
]
# print(crisp_results)


def normalize_weight_matrix(matrix):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    normalization_factors = np.empty(cols, dtype=np.double)
    for j in range(cols):
        # Calculate the sum of squares for the j-th column
        sum_of_squares = np.sum(matrix[:, j] ** 2)
        normalization_factors[j] = np.sqrt(sum_of_squares)

    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = (
                matrix[i, j] / normalization_factors[j]
            ) * weights_in_criterias_order[j]

    return matrix


weighted_normalized_results = normalize_weight_matrix(crisp_results)
# print(weighted_normalized_results)

positive_ideal_solution = np.max(weighted_normalized_results, axis=0)
negative_ideal_solution = np.min(weighted_normalized_results, axis=0)
positive_distance = np.sqrt(
    np.sum((weighted_normalized_results - positive_ideal_solution) ** 2, axis=1)
)
negative_distance = np.sqrt(
    np.sum((weighted_normalized_results - negative_ideal_solution) ** 2, axis=1)
)
# print(positive_distance, negative_distance)

closeness_coefficient = negative_distance / (positive_distance - negative_distance)
# print(closeness_coefficient)

ranked_alternatives = np.argsort(closeness_coefficient)[::-1]
ranked_alternatives = [alternates[i] for i in ranked_alternatives]
print(ranked_alternatives)
