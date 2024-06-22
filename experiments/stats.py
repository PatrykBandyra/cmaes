import pandas as pd
import csv

# Define the path to the CSV file
file_path = 'maes_ipop_default/ipop_log.csv'

# Initialize lists to hold the processed data
function_names = []
dimensions = []
initial_populations = []
final_populations = []
stagnation_types = []

# Read and process the CSV file manually
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 5:
            continue  # Skip malformed rows
        final_pop = int(row[0])
        func_name = row[1]
        dim = int(row[2])
        initial_pop = int(row[3])
        stagnation = row[4]

        function_names.append(func_name)
        dimensions.append(dim)
        initial_populations.append(initial_pop)
        final_populations.append(final_pop)
        stagnation_types.append(stagnation)

# Create a DataFrame from the processed data
data = pd.DataFrame({
    'Funkcja': function_names,
    'Wymiar': dimensions,
    'Początkowa liczebność populacji': initial_populations,
    'Końcowa liczebność populacji': final_populations,
    'Rodzaj stagnacji': stagnation_types
})

# Extracting statistics for the report
stagnation_stats = data.groupby(['Funkcja', 'Rodzaj stagnacji', 'Wymiar']).size().reset_index(name='Ilość wykryć')

print(stagnation_stats)

# Save the statistics to a new CSV file
stagnation_stats.to_csv('stagnation_statistics.csv', index=False)