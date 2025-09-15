datadict = {'BackGround' :0, 
            'Bleed-Subdural': 1,  
            'Scalp-Hematoma' :2, 
            'Bleed-Others':3, 
            'Bleed-Intraventricular':4,  
            'Bleed-Epidural': 5 , 
            'Bleed-Contusion': 6, 
            'Bleed-Hematoma' :7, 
            'Bleed-Subarachnoid': 8}


newDatadict = {
    'BackGround': 0,
    'Bleed-Subdural': 1,
    'Scalp-Hematoma': 2,
    'Bleed-Others': 3,
    'Bleed-Intraventricular': 4,
    'Bleed-Epidural': 5,}

TrainingDir = '/home/omen/Downloads/label_192'

batch_size = 2
num_workers = 8
pin_memory = True
LEARNING_RATE = 0.0005
num_epochs = 300

# IMAGE_HEIGHT = 128
# IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128