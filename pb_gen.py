import os, argparse
import sys
import six
import shutil
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from six.moves import zip, range, filter, urllib, BaseHTTPServer

from LP_model import squeeze_net

def create_inference_graph(batch_size=None, use_new_decoder=False):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size,256,256,3], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # Calculate the logits of the batch using BiRNN
    #logits = BiRNN(input_tensor, tf.to_int64(seq_length) if FLAGS.use_seq_length else None, no_dropout)
    with tf.variable_scope('squeezenet'):
        logits = squeeze_net(input_tensor, False, False)

    # Beam search decode the batch
    #decoder = decode_with_lm if use_new_decoder else tf.nn.ctc_beam_search_decoder

    #decoded, _ = decoder(logits, seq_length, merge_repeated=False, beam_width=FLAGS.beam_width)
    #decoded = tf.convert_to_tensor(
     #   [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

    return ({'input': input_tensor},{'outputs': logits})


global session_config
session_config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=FLAGS.log_placement)
checkpoint_dir = '/ayu-disk/sqCatDog/ckpt'
remove_export = False
export_dir = '/ayu-disk/sqCatDog/ckpt'


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    #log_info('Exporting the model...')
    with tf.device('/cpu:0'):
        tf.reset_default_graph()
        session = tf.Session(config=session_config)
        inputs, outputs = create_inference_graph()
        # TODO: Transform the decoded output to a string
        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())
        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path
        if remove_export:
            if os.path.isdir(export_dir):
                #log_info('Removing old export')
                print('Remove old export')
                shutil.rmtree(export_dir)
        try:
            output_graph_path = os.path.join(export_dir, 'jeans1.pb')
            if not os.path.isdir(export_dir):
                os.makedirs(export_dir)
            # Freeze graph
            freeze_graph.freeze_graph_with_def_protos(
                input_graph_def=session.graph_def,
                input_saver_def=saver.as_saver_def(),
                input_checkpoint=checkpoint_path,
                output_node_names=','.join(node.op.name for node in six.itervalues(outputs)),
                restore_op_name=None,
                filename_tensor_name=None,
                output_graph=output_graph_path,
                clear_devices=False,
                initializer_nodes='')
            #log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError as e:
            print('Lolwa')
            #log_error(str(e))

export()