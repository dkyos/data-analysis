keyword_clip_dict = dict(keyword_clip)
keyword_dict = dict(zip(keyword_clip_dict.keys(), range(len(keyword_clip_dict))))

#공백과 미학습 단어 처리를 위한 사전 정보 추가  
keyword_dict['_PAD_'] = len(keyword_dict)
keyword_dict['_UNK_'] = len(keyword_dict) 

#키워드를 역추적하기 위한 사전 생성 
keyword_rev_dict = dict([(v,k) for k, v in keyword_dict.items()])

#리뷰 시퀀스 단어수의 중앙값 +5를 max 리뷰 시퀀스로 정함... 
max_seq =np.median([len(k) for k in keywords]) + 5

def encoding_and_padding(corp_list, dic, max_seq=50):
    from keras.preprocessing.sequence import pad_sequences
    coding_seq = [ [dic.get(j, dic['_UNK_']) for j in i]  for i in corp_list ]
    #일반적으로 리뷰는 마지막 부분에 많은 정보를 포함할 가능성이 많아 패딩은 앞에 준다. 
    return(pad_sequences(coding_seq, maxlen=max_seq, padding='pre', truncating='pre',value=dic['_PAD_']))

train_x = encoding_and_padding(keywords, keyword_dict, max_seq=int(max_seq))

train_y = tbl['label']

train_x.shape, train_y.shape

from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.optimizers import *
from keras.callbacks import *
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import keras.backend as K

x_dim = train_x.shape[1]

inputs = Input(shape=(train_x.shape[1],), name='input')

embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=50,name='embedding')(inputs)

conv0 = Conv1D(32, 1, padding='same')(embeddings_out)
conv1 = Conv1D(16, 2, padding='same')(embeddings_out)
conv2 = Conv1D(8, 3, padding='same')(embeddings_out)

pool0 = AveragePooling1D()(conv0)
pool1 = AveragePooling1D()(conv1)
pool2 = AveragePooling1D()(conv2)

concat_layer = concatenate([pool0, pool1, pool2],axis=2)

bidir =Bidirectional(GRU(10, recurrent_dropout=0.2, dropout=0.2))(concat_layer)

out = Dense(1,activation='sigmoid')(bidir)

model = Model(inputs=[inputs,], outputs=out)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
hist = model.fit(x=train_x,y=train_y, batch_size=100, epochs=10, validation_split=0.1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

tbl_test = pd.read_csv("ratings_test.txt",sep='\t')
parsed_text = [mecab.morphs(str(i).strip()) for i in tbl_test['document']]
test_x = encoding_and_padding(parsed_text, keyword_dict, max_seq=int(max_seq))
test_y = tbl_test['label']
test_x.shape, test_y.shape
prob = model.predict(test_x)

#개인적으로 모형 퍼포먼스 시각화는 R을 주로 활용하는편이다... ^^;..AUC 0.93정도로 꽤 좋은 성능이다.
%reload_ext rpy2.ipython
%%R -i test_y -i prob
require(pROC)
plot(roc(test_y, prob), print.auc=TRUE)

def grad_cam_conv1D(model, layer_nm, x, sample_weight=1,  keras_phase=0):
    import keras.backend as K
    import numpy as np
    
    #레이어 이름에 해당되는 레이어 정보를 가져옴 
    layers_wt = model.get_layer(layer_nm).weights
    layers = model.get_layer(layer_nm)
    layers_weights = model.get_layer(layer_nm).get_weights()
    
    #긍정 클래스를 설명할 수 있게 컨볼루션 필터 가중치의 gradient를 구함  
    grads = K.gradients(model.output[:,0], layers_wt)[0]
    
    #필터별로 가중치를 구함 
    pooled_grads = K.mean(grads, axis=(0,1))
    get_pooled_grads = K.function([model.input,model.sample_weights[0], K.learning_phase()], 
                         [pooled_grads, layers.output[0]])
    
    pooled_grads_value, conv_layer_output_value = get_pooled_grads([[x,], [sample_weight,], keras_phase])
    #다시한번 이야기 하지만 loss를 줄이기 위한 학습과정이 아니다... 
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    return((heatmap, pooled_grads_value))

# test셋에서 90번째 인덱스에 해당하는 데이터를 시각화 해본다. 
idx = 90
prob[idx], tbl_test.iloc[idx], test_y[idx]
(array([ 0.01689465], dtype=float32),
hm, graded = grad_cam_conv1D(model, 'conv1d_1', x=test_x[idx])
hm_tbl = pd.DataFrame({'heat':hm, 'kw':[keyword_rev_dict[i] for i in test_x[idx] ]})
%%R -i hm_tbl
library(ggplot2)
library(extrafont)

ggplot(hm_tbl, aes(x=kw, y=heat)) + geom_bar(stat='identity') + theme_bw(base_family = 'UnDotum')


 

 


