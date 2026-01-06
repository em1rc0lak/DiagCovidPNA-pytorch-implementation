import os

class Config:
    IMG_SIZE = 150
    IMG_WIDTH = 150
    IMG_HEIGHT = 150
    CHANNELS = 3
    INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

    LR_SINGLE = 0.0003
    BATCH_SIZE_SINGLE = 32
    EPOCHS_SINGLE = 100

    LR_HYBRID = 0.00003
    BATCH_SIZE_HYBRID = 16
    EPOCHS_HYBRID = 150

    # Data Paths
    TRAIN_DIR = os.path.join("data", "Train")
    VALIDATION_DIR = os.path.join("data", "Valid")
    TEST_DIR = os.path.join("data", "Test")
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Classes
    CLASSES_4 = ['COVID-19', 'Normal', 'Bacterial_PNA', 'Viral_PNA']
    CLASSES_3 = ['COVID-19', 'Normal', 'Pneumonia'] 
    CLASSES_2 = ['Bacterial_PNA', 'Viral_PNA']     

    MODELS = {
    'model1': {
        'architecture': 'xception',
        'num_classes': 4,
        'use_contrast': False,
        'use_augmentation': True,
        'use_transfer_learning': True
    },
    'model2': {
        'architecture': 'xception',
        'num_classes': 4,
        'use_contrast': True,
        'use_augmentation': False,
        'use_transfer_learning': True
    },
    'model3': {
        'architecture': 'xception',
        'num_classes': 4,
        'use_contrast': False,
        'use_augmentation': False,
        'use_transfer_learning': True
    },
    'model4': {
        'architecture': 'xception',
        'num_classes': 4,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': False
    },
    'model5': {
        'architecture': 'xception',
        'num_classes': 4,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': True
    },
    'model6': {
        'architecture': 'inception_v3',
        'num_classes': 4,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': True
    },
    'model7': {
        'architecture': 'inception_resnet_v2',
        'num_classes': 4,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': True
    },

    'model8': {
        'architecture': 'xception',
        'num_classes': 2,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': True
    },
    'model9': {
        'architecture': 'xception',
        'num_classes': 3,
        'use_contrast': True,
        'use_augmentation': True,
        'use_transfer_learning': True
    }
    
}

    @staticmethod
    def get_model_config(model_name):
        """Get configuration for a specific model."""
        return Config.MODELS.get(model_name, Config.MODELS['model5'])