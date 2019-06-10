import tensorflow as tf
seq_len=[1,3,5]
input_mask = tf.sequence_mask(lengths=seq_len,dtype=tf.int32)
with tf.Session() as sess:
    print(sess.run(input_mask))