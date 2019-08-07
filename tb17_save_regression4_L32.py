### the way to evaluate the irregular domain size, see figure 6 the publication: https://aip.scitation.org/doi/10.1063/1.4973574
###
import tensorflow as tf
from PIL import Image
import glob
from glob import glob
import os

## length and width of images
l_img , w_img = 64 , 64

#tf.gfile.Remove('log-tb13')


#nn_constst = 8

#
def image2tfrecord(image_list,label_list):
    global l_img , w_img
    len2 = len(image_list)
    print("len=",len2)
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for i in range(len2):
        # read original file
        image = Image.open(image_list[i])
        image = image.convert("L")
        image = image.resize((l_img,w_img))
        # converting to bytes
        image_bytes = image.tobytes()
        # create dict
        features = {}
        # save images in bytes
        features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        # express labels by integrals
        features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(label_list[i])*float(label_list[i])]))
        # combine all features into one
        tf_features = tf.train.Features(feature=features)
        # converting to examples
        tf_example = tf.train.Example(features=tf_features)
        # serilize the samples
        tf_serialized = tf_example.SerializeToString()
        # write data into tfrecords
        writer.write(tf_serialized)
    writer.close()



def pares_tf(example_proto):
    global l_img , w_img
    global nn_constst
    # define dict
    dics = {}
    dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.float32)
    dics['image'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    # parse single sample
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    image = tf.decode_raw(parsed_example['image'],out_type=tf.uint8)
    image = tf.reshape(image,shape=[l_img*w_img])
    # normalization
    image = tf.cast(image,tf.float32)*(1./255)-0.5
#    image = tf.cast(image,tf.float32)
    label = parsed_example['label']
    label = tf.cast(label,tf.int64)
#    label = tf.one_hot(label, depth=nn_constst, on_value=1)
    label = tf.cast( label, dtype=tf.float32 )
    return image,label










train_image_list = [] 
train_image_label_list = []


cwd = os.getcwd()
paths = glob('*/')

#### obtain the directory and place the samples into the list
for path in paths :
    i = 0
    train_image_list.append ( glob( path+ "/*.jpg" ) )
#    print(type(train_image_list))
#    print(train_image_list)
#    print(path)
    path_new = path.replace("/","")
    print(path_new)
    aa = float( path_new )
    train_image_label_list.append( [aa] * len(  glob( path+ "/*.jpg" ) ) )
    
#    print( train_image_label_list ) 
    i = i + 1
#    

  
from itertools import chain 
print( train_image_list )
train_image_list = list(chain.from_iterable(train_image_list)) 
train_image_label_list = list(chain.from_iterable(train_image_label_list)) 
#

#print( train_image_label_list )

image2tfrecord(train_image_list,train_image_label_list)
dataset = tf.data.TFRecordDataset(filenames=['train.tfrecords'])
dataset = dataset.map(pares_tf).repeat(4)
size_batch = 1
dataset = dataset.shuffle(400).batch(size_batch)

print(dataset)

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

# define the figure size
x = tf.placeholder(dtype=tf.float32,shape=[None,l_img*w_img],name="x")
tf.add_to_collection("input",x)

# define the dataset of label
y_ = tf.placeholder(dtype=tf.float32,shape=[size_batch],name="y_")

print(x,y_)
is_training = tf.placeholder(tf.bool, name="training")

print(x,y_)



# convert 3d data to 2d data
image = tf.reshape(x,shape=[-1,l_img,w_img,1])
#
print(image.shape)

def cnn_layer( n_c , n_kernel , input_conv , n_layer  ):
    weight = tf.Variable(initial_value=tf.random_normal(shape=[4,4,n_c,n_kernel],stddev=0.1,dtype=tf.float32,name="weight"+str(n_layer)))
    bias= tf.Variable(initial_value=tf.zeros(shape=[n_kernel]))
#    bias = tf.constant( 0.0 , shape=[n_kernel] )
    conv = tf.nn.conv2d(input=input_conv,filter=weight,strides=[1,1,1,1],padding="SAME",name="conv"+str(n_layer))
    # shape={None,64,64,32}
    input_act = tf.nn.bias_add(conv,bias)
#    input_act_bn = tf.layers.batch_normalization( conv ,  training = False  )
    relu = tf.nn.relu( input_act , name="relu"+str(n_layer) )
    pool = tf.nn.max_pool(value=relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    return pool


### input 6 layers of CNN
cc = 1
kk = 32
input_cnn = image
for ii in range(32) :
    pool = cnn_layer( cc , kk , input_cnn , ii+1 )
    cc = kk
    kk = 32 * (ii+2)
    input_cnn = pool
    


#FC1
n_fc1 = 32
k_fc1 = cc
input_fc = pool
w_fc1 = tf.Variable(initial_value=tf.random_normal(shape=[k_fc1,n_fc1],stddev=0.1,dtype=tf.float32,name="w_fc1"))
b_fc1 = tf.Variable(initial_value=tf.zeros(shape=[n_fc1]))

# vip, conduct reshape 
input_fc1 = tf.reshape(input_fc,shape=[-1,k_fc1],name="input_fc1")
input_act = tf.nn.bias_add( value=tf.matmul(input_fc1,w_fc1) , bias = b_fc1 )
## batch normalization
fc1 = tf.nn.relu( input_act )

input_fc2 = fc1
#input_fc2 = tf.layers.batch_normalization( fc1 , training = True )
w_fc2 = tf.Variable( initial_value=tf.random_normal(shape=[ n_fc1 , 1 ] , stddev=0.1 , dtype=tf.float32 , name="w_fc2"))
b_fc2 = tf.Variable( initial_value=tf.zeros(shape=[1]) )

y = tf.nn.bias_add( value=tf.matmul(input_fc2,w_fc2) , bias=b_fc2 )

#
tf.add_to_collection('output', y)


#


# define loss 
cost = tf.reduce_mean( tf.square( y_ - y ) )


##### tensorboards 
tf.summary.scalar('cost', cost)


# define the solver
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    solver = tf.train.AdamOptimizer(learning_rate=0.00002).minimize(loss=cost)

####
mse = tf.metrics.mean_squared_error( labels=y_, predictions=y , name = "validation_metrics_var_scope" )

# validation metric init op
validation_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="validation_metrics_var_scope")
validation_metrics_init_op = tf.variables_initializer(var_list=validation_metrics_vars, name='validation_metrics_init')

# Initialization
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
print(x)
print(image)
print(y)
#
#



##########################################################################
#### train the nn

saver = tf.train.Saver()
#sess = tf.Session( config=tf.ConfigProto(device_count = {'GPU': 0 , 'CPU': 1},   inter_op_parallelism_threads=0, intra_op_parallelism_threads=0, log_device_placement=True) )
sess = tf.Session( )

merge_op = tf.summary.merge_all()                       # operation to merge all summary

writer = tf.summary.FileWriter('../log-tb17-4-L32/', sess.graph)     # write to file




sess.run( [init_global , init_local , validation_metrics_init_op ] )

for i in range(1):
    print("start")
    
    i = 0
    try:
        while True:
            image,label = sess.run(fetches=next_element)

            _ , rs , predic_out , mse_  = sess.run(  [ solver , merge_op , y , mse ], feed_dict={x:image, y_:label  })

            print(i, "label=" , label , "predic_out=" , predic_out , "file" )

            save_path = saver.save(sess,"tb17_save_regression4_L32", global_step=i)      

            i = i + 1
            
    except tf.errors.OutOfRangeError:
        print("end!")
        



sess.close()
writer.close()

