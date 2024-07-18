# Given parameters
current_size_kb = 5  # in KB
current_params = 282441
target_params = 3000000000  # 175 billion parameters

# Calculate the ratio of parameters to size
ratio = current_params / current_size_kb

# Calculate the required size in KB for the target number of parameters
required_size_kb = target_params / ratio

# Convert the required size to MB
required_size_mb = required_size_kb / 1024  # KB to MB


print(required_size_mb)
