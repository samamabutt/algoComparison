#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wavio


def label_convert_class(class_id):
    
    if class_id == 'air_conditioner':
       return 0
    elif class_id == 'car_horn':
        return 1
    elif (class_id == 'children_playing'):
        return 2
    elif (class_id == 'dog_bark'):
        return 3
    elif (class_id == 'drilling'):
        return 4
    elif (class_id == 'engine_idling'):
        return 5
    elif (class_id == 'gun_shot'):
        return 6
    elif (class_id == 'jackhammer'):
        return 7
    elif (class_id == 'siren'):
        return 8
    elif (class_id == 'street_music'):
        return 9
    





image_size = 28 # width and length

no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


train_data = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")


train_paths = np.array(train_data['slice_file_name'], dtype=pd.Series)
train_folders = np.array(train_data['fold'],dtype=pd.Series)
train_class_ids = np.array(train_data['classID'],dtype=pd.Series)

labeled_sound = []


'''
for i in range(len(train_class)):
    
    labeled_sound.append(label_convert_class(train_class[i]))
    
'''   
    
# str(train_ids[8])

    
    
#fs, data = wavfile.read('train/Train/' + '17'+ '.wav')
    
#data = wavio.read('train/Train/' + '105'+ '.wav')
    
import soundfile as sf
import pickle

my_list = []
id_list = []


'''


for i in range(len(train_paths)):
    
    data, samplerate = sf.read('UrbanSound8K/audio/' + 'fold'+ str(train_folders[i])+ '/'+ str(train_paths[i]))
    num_py_array = np.array(data)
    flaten_sound = num_py_array.flatten()
    my_list.append(flaten_sound)
    id_list.append(train_class_ids[i])
    
    if (i % 500) == 0:
        with open("pickled_urban" + str(i / 500) + ".pkl", "bw") as fh:
            data = (my_list, id_list)
            pickle.dump(data, fh)
        my_list = []
        id_list = []


'''



data_list = []
id_final_list = []



for i in range(len(train_paths)):

    
    if (i % 500) == 0:
       # with open("pickled_urban" + str(i / 500) + ".pkl", "br") as fh:
        with open("pickled_urban" + str(i / 500) + ".pkl", "br") as fh:
            data = pickle.load(fh)
            first = data[0]
            second = data[1]
            
            for z in range(len(first)):
                data_list.append(first[z])
                id_final_list.append(second[z])
                


np_arr_final = np.array(data_list, copy = True)

min = 10000000000
max = 0


sum_value = 0.0000000000000000000000000
different_lengths = []

filtered_sounds = []
filtered_ids = []



for r in range(len(np_arr_final)):
 #   if (id_final_list[r] == 5):
 
 
    #  if (len(np_arr_final[r]) == 352800):
      #    filtered_sounds.append(np_arr_final[r])
     #     filtered_ids.append(id_final_list[r])
          
    
    
        length_found = len(np_arr_final[r])
            
        different_lengths.append(length_found)
            
        sum_value = sum_value + len(np_arr_final[r])
        if (len(np_arr_final[r]) > max):
            max = len(np_arr_final[r])
        if (len(np_arr_final[r]) < min):
            min = len(np_arr_final[r])
                
                
 
        


from collections import Counter         

print(min, '---', max )
print(sum_value/len(np_arr_final))
#print(different_lengths)

print(Counter(different_lengths))


print(len(filtered_sounds))
print(len(filtered_ids))
    
'''

print('Done')

#sf.write('new_file.wav',new_flaten_sound, samplerate)

#print(new_flaten_sound.shape)




#
#test_imgs = np.asfarray(test_data[:, 1:]) / fac
#train_labels = np.asfarray(train_data[:, :1])
#test_labels = np.asfarray(test_data[:, :1])





'''
'''
import pickle
with open("pickled_mnist.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, fh)
    
'''

import pickle
with open("pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
    
train_imgs_primary = data[0]
test_imgs = data[1]
train_labels_primary = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size




##print(train_imgs[0])
##print('------------')
##print(trans_matrix[0])
##intermediate_matrix = np.linalg.inv(np.dot(trans_matrix , train_imgs))
#
##final_matrix = intermediate_matrix * trans_matrix * train_labels
#
#
#inverse_matrix = np.linalg.inv(train_imgs)
#
#print(train_imgs[0])
#print('------------')
#print(inverse_matrix[0])


'''
trans_matrix = train_imgs.transpose()
se = np.dot(trans_matrix , train_imgs)
te = np.linalg.pinv(se)
de = np.dot(trans_matrix,train_labels)
le = np.dot(te,de)

final_result = np.dot(train_imgs[0], le)

print(final_result)

'''

'''
del filteresounds
del id_final_list
'''

import gc
gc.collect()


#np_arr_final = []
#data_list = []
#id_final_list = []
#filtered_sounds = []
#filtered_ids = []

# del np_arr_final
# del data_list
# del id_final_list
# del filtered_sounds
# del filtered_ids 


train_sounds = np.array(filtered_sounds[:50], copy = True)
train_ids = np.array(filtered_ids[:50], copy = True)


print(len(train_sounds))
print(len(train_ids))

print(len(train_ids))

############################################################

trans_matrix = train_sounds.transpose()
se = np.matmul(trans_matrix , train_sounds)
te = np.linalg.pinv(se)
de = np.matmul(trans_matrix, train_ids)
le = np.matmul(te,de)


sample = train_sounds[1]
le = le.reshape(784,)

final_result = np.dot(le, sample)

#print(final_result)

   
#print(test_labels[8])



learning_rate = 0.0001

total_val = 0


'''


for j in range (1000):
    for i in range(352800):
        final_error = 0.00000000000
        for s in range(100):
            error = 0.000000
            inter_result = np.dot(le, train_sounds[s])
            expec_result = train_ids[s][0]
            error = expec_result - inter_result
            error = error * train_sounds[s][i]
            final_error = final_error + error
    
        total_val = learning_rate * final_error
        le[i] = le[i] + total_val
'''       



final_result = np.dot(le, sample)
print(train_ids[1])
print(final_result)





#################### perceptron array #############################


perceptron_array = [0 for i in range(352800)]



learning_rate = 0.0001

total_val = 0





for j in range (100):
    for i in range(352800):
        final_error = 0.00000000000
        for s in range(50):
            error = 0.000000
            inter_result = np.dot(perceptron_array, train_sounds[s])
            expec_result = train_ids[s]
            error = expec_result - inter_result
            error = error * train_sounds[s][i]
            final_error = final_error + error
    
        total_val = learning_rate * final_error
        perceptron_array[i] = perceptron_array[i] + total_val
        




final_result = np.dot(perceptron_array, sample)
print(train_ids[1])
print(final_result)






