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
beneficials = [True, True, True, True, True, False, True, False, True]

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

crisp_results = np.array(
    [
        [criteria.de_nutrosophication() for criteria in alternative]
        for alternative in aggregated_results
    ],
    dtype=np.double,
)

# criterias = ["Price To Cost", "Storage Space", "Camera", "Looks"]
# beneficials = [False, True, True, True]

# alternates = ["Mobile 1", "Mobile 2", "Mobile 3", "Mobile 4", "Mobile 5"]

# weights_in_criterias_order = np.array(
#     [
#         0.25,
#         0.25,
#         0.25,
#         0.25,
#     ]
# )


# crisp_results = np.array(
#     [
#         [250, 16, 12, 5],
#         [200, 16, 8, 3],
#         [300, 32, 16, 4],
#         [275, 32, 8, 4],
#         [225, 16, 16, 2],
#     ],
#     dtype=np.double,
# )
# crisp_results = np.array(
#     [
#         [0.536, 0.564, 0.763, 0.943, 0.728, 0.734, 0.800, 0.827, 0.910],
#         [0.564, 0.694, 0.729, 0.903, 0.808, 0.394, 0.736, 0.247, 0.910],
#         [0.698, 0.266, 0.431, 0.778, 0.789, 0.258, 0.760, 0.774, 0.881],
#         [0.724, 0.298, 0.742, 0.778, 0.467, 0.734, 0.506, 0.774, 0.910],
#         [0.754, 0.298, 0.854, 0.720, 0.267, 0.258, 0.884, 0.736, 0.936],
#     ]
# )
# print(crisp_results)


def normalize_weight_matrix(matrix):
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

max_values = np.max(weighted_normalized_results, axis=0)
min_values = np.min(weighted_normalized_results, axis=0)

positive_ideal_solution = [
    max_values[idx] if benefit else min_values[idx]
    for idx, benefit in enumerate(beneficials)
]
negative_ideal_solution = [
    min_values[idx] if benefit else max_values[idx]
    for idx, benefit in enumerate(beneficials)
]
# print(positive_ideal_solution)
# print(negative_ideal_solution)

positive_distance = np.sqrt(
    np.sum((weighted_normalized_results - positive_ideal_solution) ** 2, axis=1)
)
negative_distance = np.sqrt(
    np.sum((weighted_normalized_results - negative_ideal_solution) ** 2, axis=1)
)

# print(positive_distance)
# print(negative_distance)

closeness_coefficient = negative_distance / (positive_distance + negative_distance)
print(closeness_coefficient)

ranked_alternatives = np.argsort(closeness_coefficient)[::-1]
ranked_alternatives = [alternates[i] for i in ranked_alternatives]
print(ranked_alternatives)
