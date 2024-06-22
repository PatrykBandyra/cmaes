import pandas as pd

import numpy as np

# Define the path to log
file_path = 'maes_ipop_default/ipop_log.csv'

# Create a dictionary to map function identifiers to their names
function_mapping = {
    'f001': 'Sphere',
    'f002': 'Ellipsoidal',
    'f003': 'Rastrigin',
    'f004': 'Büche-Rastrigin',
    'f005': 'Linear Slope',
    'f006': 'Attractive Sector',
    'f007': 'Step Ellipsoidal',
    'f008': 'Rosenbrock',
    'f009': 'Rosenbrock Rotated',
    'f010': 'Ellipsoidal Rotated',
    'f011': 'Discus',
    'f012': 'Bent Cigar',
    'f013': 'Sharp Ridge',
    'f014': 'Different Powers',
    'f015': 'Weierstrass',
    'f016': 'Schaffers F7',
    'f017': 'Schaffers F7 mod. 2',
    'f018': 'Griewank-Rosenbrock',
    'f019': 'Schwefel',
    'f020': 'Gallagher\'s 101-me Peaks',
    'f021': 'Gallagher\'s 21-hi Peaks',
    'f022': 'Katsuura',
    'f023': 'Lunacek bi-Rastrigin',
    'f024': 'Non-Continuous Rastrigin'
}

# Create a dictionary for short stagnation reasons
stagnation_reasons_mapping = {
    "No progress in optimization": "NoProgress",
    "All fitness values NaN or infinite": "NaNOrInf",
    "Covariance matrix condition too high": "CovCondition",
    "Standard deviation too small": "StdDevSmall",
    "No effect axis": "NoEffectAxis",
    "No effect coordinate": "NoEffectCoord"
}

# Initialize lists to hold the processed data
function_names = []
dimensions = []
initial_populations = []
final_populations = []
stagnation_types = []

# Read and process the CSV file manually
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = pd.read_csv(csvfile)
    for _, row in reader.iterrows():
        if len(row) < 5:
            continue  # Skip malformed rows
        final_pop = int(row[0])
        func_name = row[1]
        dim = int(row[2])
        initial_pop = int(row[3])
        stagnation = row[4]

        # Extract function identifier
        func_id = func_name.split('_')[1]
        # Map function identifier to full name
        func_full_name = function_mapping.get(func_id, 'Unknown') + " " + func_id

        # Map stagnation reason to its short version
        stagnation_short = stagnation_reasons_mapping.get(stagnation.split(":")[0], 'Unknown')

        function_names.append(func_full_name)
        dimensions.append(dim)
        initial_populations.append(initial_pop)
        final_populations.append(final_pop)
        stagnation_types.append(stagnation_short)

# Create a DataFrame from the processed data
data = pd.DataFrame({
    'Funkcja': function_names,
    'Wymiar': dimensions,
    'Początkowa liczebność populacji': initial_populations,
    'Końcowa liczebność populacji': final_populations,
    'Rodzaj stagnacji': stagnation_types
})

# Group by function, stagnation type, and dimension, then calculate the mean population sizes
mean_pop_sizes = data.groupby(['Funkcja', 'Rodzaj stagnacji', 'Wymiar']).agg({
    'Początkowa liczebność populacji': 'mean',
    'Końcowa liczebność populacji': 'mean'
}).reset_index()

# Renaming columns for clarity
mean_pop_sizes.columns = ['Funkcja', 'Rodzaj stagnacji', 'Wymiar', 'Średnia początkowa liczebność populacji',
                          'Średnia końcowa liczebność populacji']

# Save the statistics to a new CSV file
mean_pop_sizes.to_csv('mean_population_sizes.csv', index=False)

print(mean_pop_sizes)