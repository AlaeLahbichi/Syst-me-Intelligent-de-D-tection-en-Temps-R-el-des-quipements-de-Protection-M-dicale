import os

x = "space"

base_path = f"/Users/alaethelegend/Downloads/ASL_Alphabet_Dataset/asl_alphabet_train/{x}"

files = sorted(os.listdir(base_path))

for filename in files:
    old_path = os.path.join(base_path, filename)

    if not os.path.isfile(old_path):
        continue

    # éviter de renommer deux fois
    if filename.startswith(x + "_"):
        continue

    new_name = f"{x}_{filename}"
    new_path = os.path.join(base_path, new_name)

    print(f"{filename} -> {new_name}")

    os.rename(old_path, new_path)

print("Renommage terminé ✅")