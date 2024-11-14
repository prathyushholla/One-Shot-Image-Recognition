# Imports 
import os 

# Path of parent dir
PARENT_DIR = "E:\code\projects\OneShotImageRecognition"

# Paths of anchor, positive and negative dataset folders
POS_PATH = os.path.join(PARENT_DIR, 'data', 'positive')
NEG_PATH = os.path.join(PARENT_DIR, 'data', 'negative')
ANC_PATH = os.path.join(PARENT_DIR, 'data', 'anchor')


if __name__ == "__main__":
    # Make the dirs 
    # Uncomment the next 4 lines if dirs are already not present 
    # os.makedirs(os.path.join(PARENT_DIR, 'data'))
    # os.makedirs(POS_PATH)
    # os.makedirs(NEG_PATH)
    # os.makedirs(ANC_PATH)

    # Move the LFW images to data/negative 
    for dir in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', dir)):
            EX_PATH = os.path.join('lfw', dir, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)

    print(f"Succesfully moved all the images to {NEG_PATH}")