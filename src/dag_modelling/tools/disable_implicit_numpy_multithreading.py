from os import environ

# Force single threaded mode. Must be done before numpy is loaded.
# Overrides variables only if they are not set.
num_threads = "1"
environment_variables = "OMP_NUM_THREADS" "OPENBLAS_NUM_THREADS" "MKL_NUM_THREADS"

for var_name in environment_variables:
    if var_name in environ:
        continue

    environ[var_name] = num_threads
