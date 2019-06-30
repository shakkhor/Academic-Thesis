from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pickle
import numpy as np # linear algebra
import pandas as pd 

model = load_model('senti.h5') 


# saving
#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

for i in range(50):
	print("input please:")
	twt = input()
	#vectorizing the tweet by the pre-fitted tokenizer instance
	twt = tokenizer.texts_to_sequences(twt)
	#padding the tweet to have exactly the same shape as `embedding_2` input
	twt = pad_sequences(twt, maxlen=250, dtype='int32', value=0)
	#print(twt)
	sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
	if(np.argmax(sentiment) == 0):
	    print("negative")
	elif (np.argmax(sentiment) == 1):
	    print("positive")




