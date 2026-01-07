from data.teeth_dataset import TeethDataset
import matplotlib.pyplot as plt
import os

# Prosta klasa udająca ustawienia
class MockOpt:
    dataroot = "."          # Kropka = bieżący folder
    phase = "train"
    max_dataset_size = float("inf")
    direction = "AtoB"
    load_size = 286         # Standard Pix2Pix
    crop_size = 256         # Standard Pix2Pix
    preprocess = "resize_and_crop"
    no_flip = False
    input_nc = 3
    output_nc = 3
    isTrain = True

if __name__ == "__main__":
    print("Sprawdzam dataset...")
    
    if not os.path.exists("wet") or not os.path.exists("dry"):
        print("❌ BŁĄD: Nie widzę folderów 'wet' i 'dry' obok skryptu!")
        exit()

    dataset = TeethDataset(MockOpt())
    print(f"✅ Sukces! Znaleziono {len(dataset)} zdjęć mokrych.")
    
    # Pobierz jedną parę
    item = dataset[0]
    print("✅ Załadowano pierwszą parę.")
    print(f"   Input (Mokry): {item['A'].shape}")
    print(f"   Target (Suchy): {item['B'].shape}")
    
    print("Gotowe do treningu!")