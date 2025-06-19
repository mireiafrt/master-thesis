# First, train a single model on the hold out real data (train+val) (can be the same as the one from utility_eval?)

# Before next step, we need the 5 synthetic GT sets (on each set, each subgroup has to match the size of the real GT)

# For each subgroup:
    # Evaluate the classifier performance on the subgroup on the REAL TEST set
    # Evaluate the classifier performance on the subgroup on the REAL GT set
    # Evaluate the classifier performance on the subgroup on EACH of the 5 SYN GT sets

# Compare difference between each TRIO of metrics (5 times, 1 for each syn GT set variation)