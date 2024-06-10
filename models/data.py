import os
import splitfolders

class DataPreprocessor():
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def preprocess_data(self, seed=13, ratio=(.8, .2)):
        print('splitting folders...')
        splitfolders.ratio(self.dataset_path,
                           output=self.output_path,
                           seed=seed,
                           ratio=ratio)

class DataPaths():
    def __init__(self, main_dir):
        self.main_dir = main_dir

    def get_train_paths(self):
        print('getting training path')
        return {
            # Add paths for other classes
            'Apple_Bad': os.path.join(self.main_dir, 'train', 'Apple_Bad'),
            'Apple_Good': os.path.join(self.main_dir, 'train', 'Apple_Good'),
            'Banana_Bad': os.path.join(self.main_dir, 'train', 'Banana_Bad'),
            'Banana_Good': os.path.join(self.main_dir, 'train', 'Banana_Good'),
            'Guava_Bad': os.path.join(self.main_dir, 'train', 'Guava_Bad'),
            'Guava_Good': os.path.join(self.main_dir, 'train', 'Guava_Good'),
            'Lime_Bad': os.path.join(self.main_dir, 'train', 'Lime_Bad'),
            'Lime_Good': os.path.join(self.main_dir, 'train', 'Lime_Good'),
            'Orange_Bad': os.path.join(self.main_dir, 'train', 'Orange_Bad'),
            'Orange_Good': os.path.join(self.main_dir, 'train', 'Orange_Good'),
            'Pomegranate_Good': os.path.join(self.main_dir, 'train', 'Pomegranate_Good'),
            'Pomegranate_Bad': os.path.join(self.main_dir, 'train', 'Pomegranate_Bad'),
        }

    def get_val_paths(self):
        print('getting validation path')
        return {
            'Apple_Bad': os.path.join(self.main_dir, 'val', 'Apple_Bad'),
            'Apple_Good': os.path.join(self.main_dir, 'val', 'Apple_Good'),
            'Banana_Bad': os.path.join(self.main_dir, 'val', 'Banana_Bad'),
            'Banana_Good': os.path.join(self.main_dir, 'val', 'Banana_Good'),
            'Guava_Bad': os.path.join(self.main_dir, 'val', 'Guava_Bad'),
            'Guava_Good': os.path.join(self.main_dir, 'val', 'Guava_Good'),
            'Lime_Bad': os.path.join(self.main_dir, 'val', 'Lime_Bad'),
            'Lime_Good': os.path.join(self.main_dir, 'val', 'Lime_Good'),
            'Orange_Bad': os.path.join(self.main_dir, 'val', 'Orange_Bad'),
            'Orange_Good': os.path.join(self.main_dir, 'val', 'Orange_Good'),
            'Pomegranate_Good': os.path.join(self.main_dir, 'val', 'Pomegranate_Good'),
            'Pomegranate_Bad': os.path.join(self.main_dir, 'val', 'Pomegranate_Bad'),
        }
