import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class TeethDataset(BaseDataset):
    """
    Loader do usuwania wody z zębów.
    Pobiera wiele klatek mokrych i paruje je z jedną klatką suchą (referencyjną).
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        # 1. Ustalamy ścieżki (zakładamy, że wet/dry są w folderze wskazanym przez --dataroot)
        self.dir_wet = os.path.join(opt.dataroot, 'wet')
        self.dir_dry = os.path.join(opt.dataroot, 'dry')

        # 2. Znajdujemy pliki
        self.wet_paths = sorted(make_dataset(self.dir_wet, opt.max_dataset_size))
        dry_paths = make_dataset(self.dir_dry, opt.max_dataset_size)

        if len(self.wet_paths) == 0 or len(dry_paths) == 0:
            raise ValueError(f"BŁĄD: Puste foldery! Sprawdź ścieżki:\n{self.dir_wet}\n{self.dir_dry}")

        # 3. Ładujemy suchy obraz RAZ do pamięci (Cache)
        self.dry_image_static = Image.open(dry_paths[0]).convert('RGB')
        
        # Zapamiętujemy ścieżkę do suchego (dla logów)
        self.dry_path_static = dry_paths[0]

    def __getitem__(self, index):
        # A. Wczytaj obraz Mokry (zmienia się co iterację)
        wet_path = self.wet_paths[index]
        wet_img = Image.open(wet_path).convert('RGB')

        # B. Pobierz obraz Suchy (z pamięci)
        dry_img = self.dry_image_static.copy()

        # C. Transformacja (Kluczowe: zsynchronizowany crop)
        # Pobieramy parametry transformacji na podstawie rozmiaru mokrego zdjęcia
        transform_params = get_params(self.opt, wet_img.size)
        
        # Tworzymy funkcję transformującą z tymi parametrami
        transform = get_transform(self.opt, transform_params, grayscale=False)

        # Aplikujemy to samo do obu zdjęć
        A = transform(wet_img)
        B = transform(dry_img)

        return {'A': A, 'B': B, 'A_paths': wet_path, 'B_paths': self.dry_path_static}

    def __len__(self):
        return len(self.wet_paths)