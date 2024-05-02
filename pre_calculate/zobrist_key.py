import numpy as np
from const import seed_value


# Set a random seed for reproducibility
np.random.seed(seed_value)

# Generate random keys
piece_keys = np.random.randint(2 ** 64 - 1, size=(2, 6, 64), dtype=np.uint64)
en_passant_keys = np.random.randint(2 ** 64 - 1, size=65, dtype=np.uint64)
castle_keys = np.random.randint(2 ** 64 - 1, size=16, dtype=np.uint64)
side_key = np.random.randint(2 ** 64 - 1, dtype=np.uint64)

# Save the arrays to files
np.save('data/piece_keys.npy', piece_keys)
np.save('data/en_passant_keys.npy', en_passant_keys)
np.save('data/castle_keys.npy', castle_keys)
np.save('data/side_key.npy', side_key)