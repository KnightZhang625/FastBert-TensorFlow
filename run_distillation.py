# coding:utf-8

import sys
import tensorflow as tf

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(MAIN_PATH))

import config as cg
from modeling import BertModel, gelu

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
	"""Compute the union of the current variables and checkpoint variables."""
	assignment_map = {}
	initialized_variable_names = {}

	name_to_variable = collections.OrderedDict()
	for var in tvars:
		name = var.name
		m = re.match("^(.*):\\d+$", name)
		if m is not None:
			name = m.group(1)
		name_to_variable[name] = var

	init_vars = tf.train.list_variables(init_checkpoint)

	assignment_map = collections.OrderedDict()
	for x in init_vars:
		(name, var) = (x[0], x[1])
		if name not in name_to_variable:
			continue
		assignment_map[name] = name
		initialized_variable_names[name] = 1
		initialized_variable_names[name + ":0"] = 1

	return (assignment_map, initialized_variable_names)

def get_specific_scope_params(scope=''):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def student_classifier(encoder_output, num_classes, scope):
	with tf.variable_scope('student_classifier_{}'.format(scope)):
		logits = tf.layers.dense(encoder_output,
														 num_classes,
														 activation=gelu,
														 name=scope,
														 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
	probalities = tf.nn.softmax(logits, axis=-1)
	
	return logits, probalities

def teacher_classfier(last_layer_output, num_classes, scope):
	with tf.variable_scope('teacher_classifier_{}'.format(scope)):
		logits = tf.layers.dense(last_layer_output,
														 num_classes,
														 activation=gelu,
														 name=scope,
														 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
	probalities = tf.nn.softmax(logits, axis=-1)

	return logits, probalities

def kl_divergence(y_true, y_pred):
	loss = y_true * np.log(y_true / (y_pred + 1e-3))
	return loss

def model_fn_builder(config, init_checkpoint=None):
	"""Returns 'model_fn' closure for Estimator."""
	def model_fn(features, labels, mode, params):
		print('*** Features ***')
		for name in sorted(features.keys()):
			tf.logging.info(' name = {}, shape = {}'.format(name, features[name].shape))

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		
		input_x = features['input_x']
		input_mask = features['input_mask']

		model = BertModel(config,
											is_training,
											input_x,
											input_mask)
		
		tvars = tf.trainable_variables()
		initialized_variable_names = {}
		if init_checkpoint:
			(assignment_map, initialized_variable_names
			) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		
		tf.logging.info("**** Trainable Variables ****")
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
											init_string)

		encoder_outputs = model.get_all_encoder_layers()
		last_encoder_outputs = encoder_outputs[-1]
		prev_encoder_outputs = encoder_outputs[:-1]

		last_outputs = teacher_classfier(last_encoder_outputs,
																		 config.num_classes,
																		 'teacher')
		
		prev_outputs = {}
		for layer_id, layer_out in enumerate(prev_encoder_outputs):
			prev_outputs[layer_id] = student_classifier(layer_out, 
																									config.num_classes, 
																									layer_id)

		if mode == tf.estimator.ModeKeys.PREDICT:
			pass
		else:
			if mode == tf.estimator.ModeKeys.TRAIN:
				last_probs = last_outputs[1]
				kl_loss = 0
				for _, value in prev_outputs.items():
					prev_probs = value[1]
					kl_loss += kl_divergence(last_probs, prev_probs)


				learning_rate = tf.train.polynomial_decay(config.LEARNING_RATE,
																					tf.train.get_or_create_global_step(),
																					config.TRAIN_STEPS,
																					end_learning_rate=config.LEARNING_LIMIT,
																					power=1.0,
																					cycle=False)
				lr = tf.maximum(tf.constant(config.LEARNING_LIMIT), learning_rate)
				
				optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
				updating_tvars = [v for v in tvars if 'bert' not in v.name]
				gradients = tf.gradients(kl_loss, updating_tvars, colocate_gradients_with_ops=True)
				clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
				train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())

				logging_hook = tf.train.LoggingTensorHook({'step': tf.train.get_global_step(),
																									'kl_loss': kl_loss,
																									'lr': learning_rate}, every_n_iter=2)
				output_spec = tf.estimator.EstimatorSpec(mode, loss=kl_loss, train_op=train_op, training_hooks=[logging_hook])





def main():
	Path(cg.SAVE_MODEL_PATH).mkdir(exist_ok=True)

	
