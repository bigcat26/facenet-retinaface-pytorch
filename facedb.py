import numpy as np

r = np.load('model_data/mobilenet_names.npy')
print(r)

r = np.load('model_data/mobilenet_face_encoding.npy')
print(r)

# d1 = {'key1':[5,10], 'key2':[50,100]}
# np.save("feat.npy", d1)

# d2 = np.load("feat.npy", allow_pickle=True)
# print(d2)

#print d1.get('key1')
#print d2.item().get('key2')