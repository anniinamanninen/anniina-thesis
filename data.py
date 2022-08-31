import numpy
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import os
#print(os.environ['PATH'])
from openslide import OpenSlide
import numpy as np

class SlidePatchDataset(Dataset):

    def __init__(self, slides_path, coords_path, splits):

        super().__init__()
        slides_path = Path(slides_path)
        slides_paths = list(slides_path.glob('**/*.svs')) #Täällä on kaikki slide-polut

        coords_path = Path(coords_path)
        coords_paths = list(coords_path.glob('**/*.h5')) #Täällä on kaikki koordinaattien polut
    
        slide_ids = list([x.stem for x in slides_paths])
        coord_ids = list([x.stem for x in coords_paths])
        matches = set(slide_ids)  &  set(splits)
        matches = list(matches)

        slide_paths_splitted = []
        coords_paths_splitted = []

        for slide, slide_id in zip(slides_paths, slide_ids): 
            if slide_id in matches: 
                slide_paths_splitted.append(slide)

        for coord, coord_id in zip(coords_paths, coord_ids): 
            if coord_id in matches: 
                coords_paths_splitted.append(coord)         

        self.coords_paths = coords_paths_splitted

        self.slide_dict={}
        for slide_path in slide_paths_splitted:
            file_name = slide_path.stem
            self.slide_dict[file_name]=slide_path

        self.coords_tot_len = 0 # This variable includes the number of patches in the whole dataset
        self.patch_sizes = [] # This variable includes the number of patches in each image
        for file in self.coords_paths: 
            coords_len = len(h5py.File(file, 'r')['coords'])
            self.patch_sizes.append(coords_len)
            self.coords_tot_len += coords_len
        
        print('Slide dictionary', self.slide_dict)

    def __len__(self):
        return(self.coords_tot_len) #Length of SlidePatchDataset

    def __getitem__(self, idx):
        
        patch_n_total = 0
        i = 0
        for path in self.coords_paths: #Looping through the coordinate paths
            patch_n_total += self.patch_sizes[i] #Tässä oletetaan, että self.patch_sizes ja self.coords_paths on samassa järjestyksessä
            n_previous_patch = patch_n_total-self.patch_sizes[i] 
            if(idx < patch_n_total and idx >= n_previous_patch):
                coords_path = path #Ollaan tämän tiedoston kohdalla
                patch_idx = idx - n_previous_patch #Täältä saadaan sen tiedoston oikea indeksi
                break
            i+=1

        coords = h5py.File(coords_path, 'r')   
        coords = tuple(coords['coords'][patch_idx]) #Get coordinates

        file_name = coords_path.stem
        slide_path = self.slide_dict[file_name]
        slide = OpenSlide(str(slide_path))
        patch_img = slide.read_region(coords, 0, (256, 256))
        patch = np.array(patch_img)

        print('slide path from get item', slide_path)
        return patch[:,:,:3], str(slide_path), coords # patch[:,:,:3] Takes the first two elements of np.array and from the third the first three
        # file_name should include the file name and the directory it belongs to (TCGA-KICH, TCGA-KIRC, TCGA-KIRP)
