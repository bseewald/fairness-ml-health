import os

# create a seed
for i in range(50):
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    print(seed)
