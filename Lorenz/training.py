from packages.utils.function_library import (load_config, training_loop)
from packages.data import LorenzDataSet
from torch.utils.data import DataLoader
import os


#TODO Create a functional training loop
#TODO Do initial Pep8 pass through code base
#TODO Implement early stopping


if __name__ == "__main__":
    for i in range(5):
        tuning_title = f"test{i+1}"
        CONFIG_PATH = f"packages/configs/{tuning_title}.yaml"
        config = load_config(CONFIG_PATH)

        # Define absolute paths to data sets
        training_dir = os.path.abspath("packages/data/training_data")
        validation_dir = os.path.abspath("packages/data/validation_data")

        # Create training and validation sets for training 
        training_dataset = LorenzDataSet(training_dir, config)
        validation_dataset = LorenzDataSet(validation_dir, 
                                        config,
                                        scalar_dict=training_dataset.scalar_dict)

        # Create dataloaders for training 
        training_loader = DataLoader(training_dataset, batch_size=config["batch_size"])
        validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"])
        print(training_dataset.targets.shape)
        print(validation_dataset.targets.shape)

        # Perform training loop
        training_loop(config, training_loader, validation_loader, tuning_title)
    



