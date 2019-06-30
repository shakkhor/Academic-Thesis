def load():
	from keras.preprocessing.text import Tokenizer
	from keras.preprocessing.sequence import pad_sequences
	from keras.models import load_model


	import numpy as np # linear algebra
	import pandas as pd 

	model = load_model('senti.h5') 

	data = pd.read_csv('./data.csv')
	data.columns = ['Class', 'Data']

	max_fatures = 20000
	tokenizer = Tokenizer(num_words=max_fatures, split=' ')
	tokenizer.fit_on_texts(data['Data'].values)

	return tokenizer