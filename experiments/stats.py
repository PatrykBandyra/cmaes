import pandas as pd
import numpy as np
import re

# Define the path to the CSV file
file_path = 'maes_ipop/ipop_log.csv'

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
    r"Stagnation detected\. No progress in optimization: range of best fitness function values is too low\.": "NoProgress",
    r"All fitness values are NaN or infinite\.": "NaNOrInf",
    r"Covariance matrix condition number is too high:": "CovCondition",
    r"Standard deviation too small: tolX =": "StdDevSmall",
    r"No effect axis: adding 0\.1-standard deviation vector in a principal axis direction does not change the mean\.": "NoEffectAxis",
    r"No effect coordinate: adding 0\.2-standard deviation in each coordinate does not change the mean\.": "NoEffectCoord"
}

# Initialize lists to hold the processed data
function_names = []
dimensions = []
initial_populations = []
final_populations = []
stagnation_types = []

# Read and process the CSV file
data_raw = pd.read_csv(file_path)

for _, row in data_raw.iterrows():
    final_pop = int(row[0])
    func_name = row[1]
    dim = int(row[2])
    initial_pop = int(row[3])
    stagnation = row[4]

    # Extract function identifier
    func_id = func_name.split('_')[1]
    # Map function identifier to full name
    func_full_name = function_mapping.get(func_id, 'Unknown') + " " + func_id

    # Map stagnation reason to its short version using regex
    stagnation_short = 'Unknown'
    for pattern, short_reason in stagnation_reasons_mapping.items():
        if re.search(pattern, stagnation):
            stagnation_short = short_reason
            break

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

# 1. Number of each type of stagnation per function
stagnation_per_function = data.groupby(['Funkcja', 'Rodzaj stagnacji']).size().reset_index(name='Ilość')

# 2. Number of each type of stagnation per dimension
stagnation_per_dimension = data.groupby(['Wymiar', 'Rodzaj stagnacji']).size().reset_index(name='Ilość')

# 3. Average population size per dimension
average_population_per_dimension = data.groupby('Wymiar').agg({
    'Początkowa liczebność populacji': 'mean',
    'Końcowa liczebność populacji': 'mean'
}).reset_index()

# Renaming columns for clarity
average_population_per_dimension.columns = ['Wymiar', 'Średnia początkowa liczebność populacji',
                                            'Średnia końcowa liczebność populacji']

# Display the DataFrames
print("Stagnation per Function:")
print(stagnation_per_function)

print("\nStagnation per Dimension:")
print(stagnation_per_dimension)

print("\nAverage Population Size per Dimension:")
print(average_population_per_dimension)

# Save the DataFrames to CSV files
stagnation_per_function.to_csv('stagnation_per_function.csv', index=False)
stagnation_per_dimension.to_csv('stagnation_per_dimension.csv', index=False)
average_population_per_dimension.to_csv('average_population_per_dimension.csv', index=False)
