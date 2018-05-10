import tensorflow as tf

def squeeze_net(x, is_training, reuse):
    input_layer = tf.reshape(x, [-1, 256, 256, 3])
    print(input_layer.shape)
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):
        with tf.variable_scope('2D-Conv-1') as scope:
            layer_1 = tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=64,
                                                      kernel_size=[3,3],stride=2,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
            m_pool_1 = tf.layers.max_pooling2d(inputs=layer_1, pool_size=[3, 3], strides=2, padding='SAME')
        print(m_pool_1.shape)    
        # Fire-1
        with tf.variable_scope('f1-sq-1') as scope:
            f1_sq_1 = tf.contrib.layers.convolution2d(inputs=m_pool_1,num_outputs=16,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f1-e-1') as scope:
            f1_e_1 = tf.contrib.layers.convolution2d(inputs=f1_sq_1,num_outputs=64,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f1-e-3') as scope:
            f1_e_3 = tf.contrib.layers.convolution2d(inputs=f1_sq_1,num_outputs=64,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_1 = tf.nn.relu(tf.concat([f1_e_1, f1_e_3],axis=3))
        print(f_1.shape)
        # Fire-2
        with tf.variable_scope('f2-sq-1') as scope:
            f2_sq_1 = tf.contrib.layers.convolution2d(inputs=f_1,num_outputs=16,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f2-e-1') as scope:
            f2_e_1 = tf.contrib.layers.convolution2d(inputs=f2_sq_1,num_outputs=64,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f2-e-3') as scope:
            f2_e_3 = tf.contrib.layers.convolution2d(inputs=f2_sq_1,num_outputs=64,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_2 = tf.nn.relu(tf.add(tf.concat([f2_e_1,f2_e_3], axis=3), f_1))
        m_pool_2 = tf.layers.max_pooling2d(inputs=f_2, pool_size=[3, 3], strides=2, padding='SAME')
        print(m_pool_2.shape)
        # Fire-3
        with tf.variable_scope('f3-sq-1') as scope:
            f3_sq_1 = tf.contrib.layers.convolution2d(inputs=m_pool_2,num_outputs=32,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f3-e-1') as scope:
            f3_e_1 = tf.contrib.layers.convolution2d(inputs=f3_sq_1,num_outputs=128,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f3-e-3') as scope:
            f3_e_3 = tf.contrib.layers.convolution2d(inputs=f3_sq_1,num_outputs=128,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_3 = tf.nn.relu(tf.concat([f3_e_1, f3_e_3],axis=3))
        print(f_3.shape)
        # Fire-4
        with tf.variable_scope('f4-sq-1') as scope:
            f4_sq_1 = tf.contrib.layers.convolution2d(inputs=f_3,num_outputs=32,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f4-e-1') as scope:
            f4_e_1 = tf.contrib.layers.convolution2d(inputs=f4_sq_1,num_outputs=128,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f4-e-3') as scope:
            f4_e_3 = tf.contrib.layers.convolution2d(inputs=f4_sq_1,num_outputs=128,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_4 = tf.nn.relu(tf.add(tf.concat([f4_e_1,f4_e_3], axis=3), f_3))
        m_pool_4 = tf.layers.max_pooling2d(inputs=f_4, pool_size=[3, 3], strides=2, padding='SAME')
        print(m_pool_4.shape)
        # Fire-5
        with tf.variable_scope('f5-sq-1') as scope:
            f5_sq_1 = tf.contrib.layers.convolution2d(inputs=m_pool_4,num_outputs=64,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f5-e-1') as scope:
            f5_e_1 = tf.contrib.layers.convolution2d(inputs=f5_sq_1,num_outputs=256,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f5-e-3') as scope:
            f5_e_3 = tf.contrib.layers.convolution2d(inputs=f5_sq_1,num_outputs=256,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_5 = tf.nn.relu(tf.concat([f5_e_1, f5_e_3],axis=3))
        print(f_5.shape)
        # Fire-6
        with tf.variable_scope('f6-sq-1') as scope:
            f6_sq_1 = tf.contrib.layers.convolution2d(inputs=f_5,num_outputs=64,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f6-e-1') as scope:
            f6_e_1 = tf.contrib.layers.convolution2d(inputs=f6_sq_1,num_outputs=256,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f6-e-3') as scope:
            f6_e_3 = tf.contrib.layers.convolution2d(inputs=f6_sq_1,num_outputs=256,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_6 = tf.nn.relu(tf.concat([f6_e_1, f6_e_3],axis=3))
        m_pool_6 = tf.layers.max_pooling2d(inputs=f_6, pool_size=[3, 3], strides=2, padding='SAME')
        print(f_6.shape)
        # Fire-7
        with tf.variable_scope('f7-sq-1') as scope:
            f7_sq_1 = tf.contrib.layers.convolution2d(inputs=m_pool_6,num_outputs=128,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f7-e-1') as scope:
            f7_e_1 = tf.contrib.layers.convolution2d(inputs=f7_sq_1,num_outputs=512,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f7-e-3') as scope:
            f7_e_3 = tf.contrib.layers.convolution2d(inputs=f7_sq_1,num_outputs=512,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_7 = tf.nn.relu(tf.concat([f7_e_1, f7_e_3],axis=3))
        print(f_7.shape)
        # Fire-8
        with tf.variable_scope('f8-sq-1') as scope:
            f8_sq_1 = tf.contrib.layers.convolution2d(inputs=f_7,num_outputs=128,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f8-e-1') as scope:
            f8_e_1 = tf.contrib.layers.convolution2d(inputs=f8_sq_1,num_outputs=512,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f8-e-3') as scope:
            f8_e_3 = tf.contrib.layers.convolution2d(inputs=f8_sq_1,num_outputs=512,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_8 = tf.nn.relu(tf.concat([f8_e_1, f8_e_3],axis=3))
        m_pool_8 = tf.layers.max_pooling2d(inputs=f_8, pool_size=[3, 3], strides=2, padding='SAME')
        print(f_8.shape)
        # Fire-9
        with tf.variable_scope('f9-sq-1') as scope:
            f9_sq_1 = tf.contrib.layers.convolution2d(inputs=m_pool_8,num_outputs=256,
                                                      kernel_size=[1,1],stride=1,activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f9-e-1') as scope:
            f9_e_1 = tf.contrib.layers.convolution2d(inputs=f9_sq_1,num_outputs=1024,
                                                      kernel_size=[1,1],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        with tf.variable_scope('f9-e-3') as scope:
            f9_e_3 = tf.contrib.layers.convolution2d(inputs=f9_sq_1,num_outputs=1024,
                                                      kernel_size=[3,3],stride=1,activation_fn=None,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
        f_9 = tf.nn.relu(tf.concat([f9_e_1, f9_e_3],axis=3))
        print(f_9.shape)


        # Final-Conv
        with tf.variable_scope('Final_Conv') as scope:
            final_conv = tf.contrib.layers.convolution2d(inputs=f_9,num_outputs=3,
                                                      kernel_size=[1,1],stride=[1,1],activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={'is_training': is_training},scope=scope,reuse=reuse)
            final_pool = tf.layers.average_pooling2d(inputs=final_conv, pool_size=[4, 4], strides=1, padding='VALID')        
        logits = tf.reshape(final_pool,[-1, 3], name='output_node')
    print(logits.shape)
    return logits

