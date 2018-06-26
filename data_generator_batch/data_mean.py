import numpy as np
import pickle
import matplotlib.image as mpimg

from scipy.misc.pilutil import imresize


txt_file = './train.txt'


print('Generating mean file...')
    
frames_channel0 = []
frames_channel1 = []
frames_channel2 = []
with open(txt_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(' ')
        img_paths = items[0]
        frame = mpimg.imread(img_paths)
        frame = imresize(frame, (60, 80, 3))
        
        frame_channel0 = frame[:,:,0]
        frame_channel1 = frame[:,:,1]
        frame_channel2 = frame[:,:,2]

        frames_channel0.append(frame_channel0)
        frames_channel1.append(frame_channel1)
        frames_channel2.append(frame_channel2)

frames_channel0 = np.array(frames_channel0, dtype=np.float32)
frames_channel1 = np.array(frames_channel1, dtype=np.float32)
frames_channel2 = np.array(frames_channel2, dtype=np.float32)

mean_channel0 = np.mean(frames_channel0)
mean_channel1 = np.mean(frames_channel1)
mean_channel2 = np.mean(frames_channel2)
     
pickle.dump({'channel0':mean_channel0, 
             'channel1':mean_channel1, 
             'channel2':mean_channel2},open("mean.pkl", "wb"))
print('Successful generate mean file.')
print('channel0: %f' % mean_channel0)
print('channel1: %f' % mean_channel1)
print('channel2: %f' % mean_channel2)       
            
