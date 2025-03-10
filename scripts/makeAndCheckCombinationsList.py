from itertools import combinations_with_replacement
from sys import argv

if (len(argv) < 5):
    print(f"Usage: {argv[0]} output_dir correct_combinations_path sequential_combinations_path id_combinations_path");
    exit(-1);
    # correctCombinationsFilePath = "../tmp/correctCombinationsList.txt"
    # sequentialCombinationsFilePath = "../tmp/sequentialCombinationsList.txt"
    # idCombinationsFilePath = "../tmp/idCombinationsList.txt"

correctCombinationsFilePath = f"{argv[1]}{argv[2]}"
sequentialCombinationsFilePath = f"{argv[1]}{argv[3]}"
idCombinationsFilePath = f"{argv[1]}{argv[4]}"

n = 16
k = 4

input_set = [i for i in range(n)]

combinations = list(combinations_with_replacement(input_set, k))

# Redundant but usefull for comparing outputs
with open(correctCombinationsFilePath, "w+") as file:
    for combination in combinations:
        combination = sorted(combination, reverse=True)
        for item in combination:
            file.write(f"{item} ")
        file.write("\n")

correctCombinations = open(correctCombinationsFilePath, "r").read().splitlines();
sequentialCombinations = open(sequentialCombinationsFilePath, "r").read().splitlines();
idCombinations = open(idCombinationsFilePath, "r").read().splitlines();

for combinationToCheck in correctCombinations:
    if (combinationToCheck not in sequentialCombinations or combinationToCheck not in idCombinations):
        print("Combination missing: ")
        print(combinationToCheck)
        exit(-1)

print("CombosCorrect test Passed")