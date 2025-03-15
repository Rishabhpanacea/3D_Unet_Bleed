datadict = {'BackGround' :0, 
            'Bleed-Subdural': 1,  
            'Scalp-Hematoma' :2, 
            'Bleed-Others':3, 
            'Bleed-Intraventricular':4,  
            'Bleed-Epidural': 5 , 
            'Bleed-Contusion': 6, 
            'Bleed-Hematoma' :7, 
            'Bleed-Subarachnoid': 8}

TrainingDir = r"C:\Users\Rishabh\Documents\pytorch-3dunet\TrainingData"

batch_size = 1
num_workers = 0
pin_memory = True
LEARNING_RATE = 1e-4
num_epochs = 2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# IMAGE_HEIGHT = 64
# IMAGE_WIDTH = 64