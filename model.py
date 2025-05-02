import logging
import numpy as np
import pdb
import tensorflow.compat.v1 as tf
from sklearn.decomposition import PCA
import tf_slim as slim
from tf_slim import layers
from memory import DKVMN
from utils import getLogger

# set logger
logger = getLogger('Deep-IRT-model-HN')


def tensor_description(var):
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description


class DeepIRTModel(object):
    def __init__(self, args, sess, name="KT"):
        self.args = args
        self.sess = sess
        self.name = name
        self.create_model()

    def create_model(self):
        self._create_placeholder()
        self._influence()
        self._create_loss()
        self._create_optimizer()
        self._add_summary()

    def _create_placeholder(self):
        logger.info("Initializing Placeholder")
        self.s_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='s_data')
        self.q_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data')
        self.qa_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
        self.label = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='label')

    def _influence(self):
        # Initialize Memory
        logger.info("Initializing Key and Value Memory")
        with tf.variable_scope("Memory"):
            init_key_memory = tf.get_variable(
                'key_memory_matrix', [self.args.memory_size, self.args.key_memory_state_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            init_value_memory = tf.get_variable(
                'value_memory_matrix', [self.args.memory_size, self.args.value_memory_state_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            
        decay_weight = []
        for distance in range(self.args.window_size):
          decay_weight.append(1/tf.math.exp(tf.cast(self.args.window_size - distance-1,tf.float32))) 
        self.decay_weight = tf.convert_to_tensor(decay_weight)


        # Boardcast value-memory matrix to Shape (batch_size, memory_size, memory_value_state_dim)
        init_value_memory = tf.tile(  # tile the number of value-memory by the number of batch
            tf.expand_dims(init_value_memory, 0),  # make the batch-axis
            tf.stack([self.args.batch_size, 1, 1])
        )
        self.vm = init_value_memory
        logger.debug("Shape of init_value_memory = {}".format(init_value_memory.get_shape()))
        logger.debug("Shape of init_key_memory = {}".format(init_key_memory.get_shape()))

        # Initialize DKVMN
        self.memory = DKVMN(
            memory_size=self.args.memory_size,
            key_memory_state_dim=self.args.key_memory_state_dim,
            value_memory_state_dim=self.args.value_memory_state_dim,
            num_pattern=self.args.num_pattern,
            delta_1=self.args.delta_1,
            delta_2=self.args.delta_2,
            rounds=self.args.rounds,
            batch_size = self.args.batch_size,
            init_key_memory=init_key_memory,
            init_value_memory=init_value_memory,
            name="DKVMN"
        )

        # Initialize Embedding
        logger.info("Initializing Q and QA Embedding")
        with tf.variable_scope('Embedding'):
            s_embed_matrix = tf.get_variable(
                's_embed', [self.args.n_skills + 1, self.args.key_memory_state_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            q_embed_matrix = tf.get_variable(
                'q_embed', [self.args.n_questions + 1, self.args.key_memory_state_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            qa_embed_matrix = tf.get_variable(
                'qa_embed', [2 * self.args.n_skills + 1, self.args.value_memory_state_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )

        self.q_embed_matrix__ = q_embed_matrix
        self.qa_embed_matrix__ = qa_embed_matrix
        # Embedding to Shape (batch size, seq_len, memory_state_dim(d_k or d_v))
        logger.info("Initializing Embedding Lookup")
        s_embed_data = tf.nn.embedding_lookup(s_embed_matrix, self.s_data)
        q_embed_data = tf.nn.embedding_lookup(q_embed_matrix, self.q_data)
        qa_embed_data = tf.nn.embedding_lookup(qa_embed_matrix, self.qa_data)
        self.s_embed_data = s_embed_data
        self.q_embed_data = q_embed_data
        self.qa_embed_data = qa_embed_data

        logger.debug("Shape of q_embed_data: {}".format(q_embed_data.get_shape()))
        logger.debug("Shape of qa_embed_data: {}".format(qa_embed_data.get_shape()))

        sliced_s_embed_data = tf.split(
            value=s_embed_data, num_or_size_splits=self.args.seq_len, axis=1
        )
        sliced_q_embed_data = tf.split(
            value=q_embed_data, num_or_size_splits=self.args.seq_len, axis=1
        )
        sliced_qa_embed_data = tf.split(
            value=qa_embed_data, num_or_size_splits=self.args.seq_len, axis=1
        )

        logger.debug("Shape of sliced_q_embed_data[0]: {}".format(sliced_q_embed_data[0].get_shape()))
        logger.debug("Shape of sliced_qa_embed_data[0]: {}".format(sliced_qa_embed_data[0].get_shape()))

        pred_z_values = list()
        pred_value_list = list()
        student_abilities = list()
        question_difficulties = list()
        memory_matrix_pre_list = list()
        attention_weight_output = list()
        attention_weight_output_pre = list()
        memory_matrix_pre_list = list()
        reuse_flag = False
        self.pre_mv_list = list()
        skill_difficulties = list()
        embed_list = list()
        logger.info("Initializing Influence Procedure")
        
        for i in range(self.args.seq_len):
            # To reuse linear vectors
            if i != 0:
                reuse_flag = True
            if self.args.window_size-1-i > 0:
                pad = self.args.window_size - 1 - i
                zero_word_emb = tf.zeros([self.args.batch_size,1, self.args.key_memory_state_dim])
            else:
                pad = 0

            # for count in range(self.args.window_size - pad - 1):
            #     window = tf.concat([sliced_q_embed_data[i-count], sliced_q_embed_data[i-count-1]], 1)
            if i>=self.args.window_size:
              window = sliced_s_embed_data[i-self.args.window_size+1:i+1]
            else:
              window = sliced_s_embed_data[0:i+1]
            new_window = None
            for j in range(len(window)):
              if j == 0:
                new_window = window[0]
              else:
                new_window = tf.concat([new_window,window[j]],axis = 1)
            
            window = new_window
            
            if pad > 0:
                for k in range(pad):
                    window = tf.concat([zero_word_emb,window],1)
            # (BS x window_size x dim)(qt-n ... qt -> qt)
            # tf.reverse(window, 1)

            # attention weight(BS x window_size)

            s = tf.squeeze(sliced_s_embed_data[i], 1)
            q = tf.squeeze(sliced_q_embed_data[i], 1)
            # print(window.shape)
            qa = tf.squeeze(sliced_qa_embed_data[i], 1)

            self.attention_weight_1 = self.memory.self_attention(s,embedded_query_vector=window,reuse = reuse_flag)


            if i == 0:
                for j in range(self.args.window_size-1):
                    zero_matrix = tf.get_variable("zero_matrix"+str(j),
                                                  [self.args.memory_size, self.args.value_memory_state_dim],
                                                  trainable=False, initializer=tf.zeros_initializer())
                    zero_matrix = tf.tile(tf.expand_dims(zero_matrix, 0), tf.stack([self.args.batch_size, 1, 1]))
                    self.pre_mv_list.append(zero_matrix)
                self.pre_mv_list.append(init_value_memory)

                for j in range(self.args.num_pattern):
                    zero_matrix1 = tf.get_variable(
                        'zero_matrix1' + str(j), [self.args.memory_size, self.args.value_memory_state_dim],
                        trainable=False,
                        initializer=tf.zeros_initializer()
                    )
                    zero_matrix1 = tf.tile(  # tile the number of value-memory by the number of batch
                        tf.expand_dims(zero_matrix1, 0),  # make the batch-axis
                        tf.stack([self.args.batch_size, 1, 1])
                    )
                    memory_matrix_pre_list.append(zero_matrix1)
            mv_list = tf.reshape(self.pre_mv_list[0],[self.args.batch_size,1,self.args.memory_size,self.args.value_memory_state_dim])
            for k in range(1,len(self.pre_mv_list)):
                # logger.debug("read_content: {}".format(self.pre_mv_list[k].get_shape))
                tmp = tf.reshape(self.pre_mv_list[k],[self.args.batch_size,1,self.args.memory_size,self.args.value_memory_state_dim])
                mv_list = tf.concat([mv_list,tmp],1)
            # mv_list bs window memorysize memorystatedim
            mv_list = tf.reshape(mv_list,[self.args.batch_size,self.args.window_size,-1])
            # self.attention_weight_1 = tf.nn.softmax(self.attention_weight_1,axis=1)
            self.attention_weight_1 = self.attention_weight_1 * tf.reshape(self.decay_weight,[1,-1])
            attention_weight_pre = self.attention_weight_1
            # x = tf.reshape(self.decay_weight,[1,-1])
            # print(x.shape)
            # self.attention_weight = self.attention_weight_1
            self.attention_weight = tf.nn.softmax(self.attention_weight_1,axis=1)
            # logger.debug("read_content: {}".format(self.attention_weight))
            attention_weight_single = self.attention_weight
            self.attention_weight = tf.reshape(self.attention_weight,[self.args.batch_size,self.args.window_size,1])

            # print(mv_list.shape)
            mv_list = layers.fully_connected(
              inputs=mv_list,
              num_outputs=self.args.value_memory_state_dim,
              scope=self.name + '/MemoryValue',
              reuse=reuse_flag,
              activation_fn=tf.tanh
            )
            self.read_content = self.attention_weight * mv_list

            self.read_content_list = tf.split(
                value=self.read_content, num_or_size_splits=self.args.window_size, axis=1
            )

            self.read_content = tf.reshape(tf.squeeze(tf.reduce_sum(self.read_content,1)),[self.args.batch_size,-1])

            # logger.debug("read_content: {}".format(self.read_content))

            # Write process, new_memory_value: Shape (batch_size, memory_size, value_memory_state_dim)
            self.new_memory_value, memory_matrix_pre = self.memory.write( qa,s, memory_matrix_pre_list, reuse=reuse_flag)

            if self.args.num_pattern > 0:
                self.memory_matrix_pre = memory_matrix_pre
                memory_matrix_pre_list.pop(0)
                memory_matrix_pre_list.append(self.memory_matrix_pre)
            # logger.debug("new_memory_value: {}".format(self.new_memory_value))
            self.pre_mv_list.pop(0)
            self.pre_mv_list.append(self.new_memory_value)

            # mastery_level_prior_difficulty = self.read_content
            mastery_level_prior_difficulty = tf.concat([self.read_content, s], 1)

            self.summary_vector = layers.fully_connected(
                inputs=mastery_level_prior_difficulty,
                num_outputs=self.args.summary_vector_output_dim,
                scope='SummaryOperation',
                reuse=reuse_flag,
                activation_fn=tf.nn.tanh
            )
            # logger.debug("summary_vector: {}".format(self.summary_vector))

            # Calculate the student ability level from summary vector
            student_ability = layers.fully_connected(
                inputs=self.summary_vector,
                num_outputs=1,
                scope='StudentAbilityOutputLayer',
                reuse=reuse_flag,
                activation_fn=None
            )

            # Calculate the question difficulty level from the question embedding
            # question_difficulty = layers.fully_connected(
            #     inputs=q,
            #     num_outputs=1,
            #     scope='QuestionDifficultyOutputLayer',
            #     reuse=reuse_flag,
            #     activation_fn=tf.nn.tanh
            # )
            question_difficulty_1 = layers.fully_connected(
                inputs=q,
                num_outputs=self.args.summary_vector_output_dim,
                scope='QuestionDifficultyOutputLayer1',
                reuse=reuse_flag,
                activation_fn=tf.nn.tanh
            )

            question_difficulty = layers.fully_connected(
                inputs=question_difficulty_1,
                num_outputs=1,
                scope='QuestionDifficultyOutputLayer',
                reuse=reuse_flag,
                activation_fn=None
            )
            skill_difficulty_1 = layers.fully_connected(
                inputs=s,
                num_outputs=self.args.summary_vector_output_dim,
                scope='QuestionskillDifficultyOutputLayer1',
                reuse=reuse_flag,
                activation_fn=tf.nn.tanh
            )

            skill_difficulty = layers.fully_connected(
                inputs=skill_difficulty_1,
                num_outputs=1,
                scope='QuestionskillDifficultyOutputLayer',
                reuse=reuse_flag,
                activation_fn=None
            )
            # Calculate the question difficulty level from the question embedding

            # Prediction
            pred_z_value = 3.0 * student_ability - question_difficulty - skill_difficulty
            pred_z_values.append(pred_z_value)
            pred_raw = tf.sigmoid(pred_z_value)
            pred_value_list.append(pred_raw)
            student_abilities.append(student_ability)
            question_difficulties.append(question_difficulty)
            # attention_weight_output.append(attention_weight_single)
            # attention_weight_output_pre.append(attention_weight_pre)
            skill_difficulties.append(tf.nn.tanh(skill_difficulty))

        self.pred_z_values = tf.reshape(
            tf.stack(pred_z_values, axis=1),
            [self.args.batch_size, self.args.seq_len]
        )
        self.pred_value_list = tf.reshape(
            tf.stack(pred_value_list, axis=1),
            [self.args.batch_size, self.args.seq_len]
        )
        self.student_abilities = tf.reshape(
            tf.stack(student_abilities, axis=1),
            [self.args.batch_size, self.args.seq_len]
        )
        self.question_difficulties = tf.reshape(
            tf.stack(question_difficulties, axis=1),
            [self.args.batch_size, self.args.seq_len]
        )
        # self.attention_weight = tf.reshape(
        #     tf.stack(attention_weight_output, axis=1),
        #     [self.args.batch_size, -1]
        # )
        # self.attention_weight_pre = tf.reshape(
        #     tf.stack(attention_weight_output_pre, axis=1),
        #     [self.args.batch_size, -1]
        # )
        self.skill_difficulties = tf.reshape(
            tf.stack(skill_difficulties, axis=1),
            [self.args.batch_size, self.args.seq_len]
        )

        logger.debug("Shape of pred_z_values: {}".format(self.pred_z_values))
        logger.debug("Shape of student_abilities: {}".format(self.student_abilities))
        logger.debug("Shape of question_difficulties: {}".format(self.question_difficulties))

    def _create_loss(self):
        logger.info("Initializing Loss Function")

        # convert into 1D
        label_1d = tf.reshape(self.label, [-1])
        pred_z_values_1d = tf.reshape(self.pred_z_values, [-1])
        student_abilities_1d = tf.reshape(self.student_abilities, [-1])
        question_difficulties_1d = tf.reshape(self.question_difficulties, [-1])
        skill_difficulties_1d = tf.reshape(self.skill_difficulties, [-1])

        # find the label index that is not masking
        index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

        # masking
        filtered_label = tf.gather(label_1d, index)
        filtered_z_values = tf.gather(pred_z_values_1d, index)
        filtered_student_abilities = tf.gather(student_abilities_1d, index)
        filtered_question_difficulties = tf.gather(question_difficulties_1d, index)
        filtered_skill_difficulties = tf.gather(skill_difficulties_1d, index)
        logger.debug("Shape of filtered_label: {}".format(filtered_label))
        logger.debug("Shape of filtered_z_values: {}".format(filtered_z_values))
        logger.debug("Shape of filtered_student_abilities: {}".format(filtered_student_abilities))
        logger.debug("Shape of filtered_question_difficulties: {}".format(filtered_question_difficulties))
        logger.debug("Shape of filtered_skill_difficulties: {}".format(filtered_skill_difficulties))
        if self.args.use_ogive_model:
            # make prediction using normal ogive model
            dist = tfd.Normal(loc=0.0, scale=1.0)

            self.pred = dist.cdf(pred_z_values_1d)
            filtered_pred = dist.cdf(filtered_z_values)
        else:
            self.pred = tf.math.sigmoid(pred_z_values_1d)
            filtered_pred = tf.math.sigmoid(filtered_z_values)

        # convert the prediction probability to logit, i.e., log(p/(1-p))
        epsilon = 1e-6
        clipped_filtered_pred = tf.clip_by_value(filtered_pred, epsilon, 1. - epsilon)
        filtered_logits = tf.log(clipped_filtered_pred / (1 - clipped_filtered_pred))

        # cross entropy loss
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=filtered_logits,
                labels=filtered_label
            )
        )

        self.loss = cross_entropy

    def _create_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
            gvs = self.optimizer.compute_gradients(self.loss)
            clipped_gvs = [(tf.clip_by_norm(grad, self.args.max_grad_norm), var) for grad, var in gvs if grad is not None]
            self.train_op = self.optimizer.apply_gradients(clipped_gvs)

    def _add_summary(self):
        tf.summary.scalar('Loss', self.loss)
        self.tensorboard_writer = tf.summary.FileWriter(
            logdir=self.args.tensorboard_dir,
            graph=self.sess.graph
        )

        model_vars = tf.trainable_variables()

        total_size = 0
        total_bytes = 0
        model_msg = ""
        for var in model_vars:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes
            model_msg += ' '.join(
                [var.name,
                 tensor_description(var),
                 '[%d, bytes: %d]' % (var_size, var_bytes)]
            )
            model_msg += '\n'
        model_msg += 'Total size of variables: %d \n' % total_size
        model_msg += 'Total bytes of variables: %d \n' % total_bytes
        logger.info(model_msg)
