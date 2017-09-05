"""
This file contains FastQA specific modules and ports
"""

import random
from tensorflow.python.ops import rnn, rnn_cell
from jtr.jack.core import *
from jtr.jack.fun import simple_model_module, no_shared_resources
from jtr.jack.tasks.xqa.shared import XQAPorts
from jtr.jack.tasks.xqa.util import unique_words_with_chars, prepare_data, char_vocab_from_vocab
from jtr.jack.tf_fun.dropout import fixed_dropout
from jtr.jack.tf_fun.embedding import conv_char_embedding_alt
from jtr.jack.tf_fun.highway import highway_network
from jtr.jack.tf_fun.rnn import birnn_with_projection
from jtr.jack.tf_fun.xqa import xqa_min_crossentropy_loss
from jtr.preprocess.batch import GeneratorWithRestart
from jtr.preprocess.map import numpify
from jtr.util import tfutil
import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention,AttentionWrapper
from tensorflow.python.ops import array_ops
import sys
from time import time
sys.path.append("/Users/apple/Downloads/project_2/jtr/jack/tasks/xqa")
import match_utils
class FastQAInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        # create character vocab + word lengths + char ids per word
        self.shared_vocab_config.config["char_vocab"] = char_vocab_from_vocab(self.shared_vocab_config.vocab)
        # Assumes that vocab and embeddings are given during creation
        self.setup()

    def setup(self):
        self.vocab = self.shared_vocab_config.vocab
        self.config = self.shared_vocab_config.config
        self.batch_size = self.config.get("batch_size", 1)
        self.dropout = self.config.get("dropout", 1)
        self._rng = random.Random(self.config.get("seed", 123))
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_vocab_config.config["char_vocab"]

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    @property
    def output_ports(self) -> List[TensorPort]:
        return [XQAPorts.emb_question, XQAPorts.question_length,
                XQAPorts.emb_support, XQAPorts.support_length,
                # char
                XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                # features
                
                XQAPorts.word_in_question,
                XQAPorts.slot_list,
                # optional, only during training
                XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                XQAPorts.keep_prob, XQAPorts.is_eval,
                # for output module
                XQAPorts.token_char_offsets,
                Ports.Input.question]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [XQAPorts.answer_span, XQAPorts.answer2question]

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]], is_eval: bool) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans, sample_ids,slot = \
            prepare_data(dataset, self.vocab, self.config.get("lowercase", False), with_answers=True,
                         max_support_length=self.config.get("max_support_length", None))

        emb_supports = np.zeros([self.batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(q_lengths), self.emb_matrix.shape[1]])

        def batch_generator():
            todo = list(range(len(q_ids)))
            if not is_eval:
                self._rng.shuffle(todo)
            while todo:
                support_lengths = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                slot_list=[]
                span2question = []
                offsets = []
                batchqs = []
                samp_ids= []

                unique_words, unique_word_lengths, question2unique, support2unique = \
                    unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab, todo[:self.batch_size])

                # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
                for i, j in enumerate(todo[:self.batch_size]):
                    support = s_ids[j]
                    for k in range(len(support)):
                        emb_supports[i, k] = self._get_emb(support[k])
                    question = q_ids[j]
                    batchqs.append(question)
                    samp_ids.append(sample_ids[j])
                    for k in range(len(question)):
                        emb_questions[i, k] = self._get_emb(question[k])
                    support_lengths.append(s_lengths[j])
                    question_lengths.append(q_lengths[j])
                    spans.extend(answer_spans[j])
                    span2question.extend(i for _ in answer_spans[j])
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])
                    slot_list.append(slot[j])
                batch_size = len(question_lengths)
                output = {
                    XQAPorts.unique_word_chars: unique_words,
                    XQAPorts.unique_word_char_length: unique_word_lengths,
                    XQAPorts.question_words2unique: question2unique,
                    XQAPorts.support_words2unique: support2unique,
                    XQAPorts.emb_support: emb_supports[:batch_size, :max(support_lengths), :],
                    XQAPorts.support_length: support_lengths,
                    XQAPorts.emb_question: emb_questions[:batch_size, :max(question_lengths), :],
                    XQAPorts.question_length: question_lengths,
                    XQAPorts.word_in_question: wiq,
                    XQAPorts.answer_span: spans,
                    XQAPorts.correct_start_training: [] if is_eval else [s[0] for s in spans],
                    XQAPorts.answer2question: span2question,
                    XQAPorts.answer2question_training: [] if is_eval else span2question,
                    XQAPorts.keep_prob: 1.0 if is_eval else 1 - self.dropout,
                    XQAPorts.is_eval: is_eval,
                    XQAPorts.token_char_offsets: offsets,
                    XQAPorts.slot_list:slot_list,
                    Ports.Input.question: batchqs,
                    Ports.Input.sample_id: samp_ids
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[XQAPorts.unique_word_chars,
                                              XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                                              XQAPorts.word_in_question,XQAPorts.slot_list, XQAPorts.token_char_offsets,
                                              XQAPorts.slot_list,
                                              Ports.Input.question])
                todo = todo[self.batch_size:]
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans,slot= prepare_data(qa_settings, self.vocab,
                                                                     self.config.get("lowercase", False),
                                                                     with_answers=False)

        unique_words, unique_word_lengths, question2unique, support2unique = \
            unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab)

        batch_size = len(qa_settings)
        emb_supports = np.zeros([batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([batch_size, max(q_lengths), self.emb_matrix.shape[1]])
        
        for i, q in enumerate(q_ids):
            for k, v in enumerate(s_ids[i]):
                emb_supports[i, k] = self._get_emb(v)
            for k, v in enumerate(q):
                emb_questions[i, k] = self._get_emb(v)

        output = {
            XQAPorts.unique_word_chars: unique_words,
            XQAPorts.unique_word_char_length: unique_word_lengths,
            XQAPorts.question_words2unique: question2unique,
            XQAPorts.support_words2unique: support2unique,
            XQAPorts.emb_support: emb_supports,
            XQAPorts.support_length: s_lengths,
            XQAPorts.emb_question: emb_questions,
            XQAPorts.question_length: q_lengths,
            XQAPorts.slot_list:slot,
            XQAPorts.word_in_question: word_in_question,
            XQAPorts.token_char_offsets: token_offsets,
            Ports.Input.question: q_ids
        }

        output = numpify(output, keys=[XQAPorts.unique_word_chars, XQAPorts.question_words2unique,
                                       XQAPorts.support_words2unique, XQAPorts.word_in_question,
                                       XQAPorts.token_char_offsets,XQAPorts.slot_list,
                                       Ports.Input.question])

        return output


fastqa_like_model_module_factory = simple_model_module(
    input_ports=[XQAPorts.emb_question, XQAPorts.question_length,
                 XQAPorts.emb_support, XQAPorts.support_length,
                 # char embedding inputs
                 XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                 XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                 # feature input
                 XQAPorts.word_in_question,
                 XQAPorts.slot_list,
                 # optional input, provided only during training
                 XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                 XQAPorts.keep_prob, XQAPorts.is_eval],
    output_ports=[XQAPorts.start_scores, XQAPorts.end_scores,
                  XQAPorts.span_prediction],
    training_input_ports=[XQAPorts.start_scores, XQAPorts.end_scores,
                          XQAPorts.answer_span, XQAPorts.answer2question],
    training_output_ports=[Ports.loss])


def fastqa_like_with_min_crossentropy_loss_factory(shared_resources, f):
    return fastqa_like_model_module_factory(shared_resources, f, no_shared_resources(xqa_min_crossentropy_loss))


# Very specialized and therefore not sharable  TF code for fast qa model.
def fatqa_model_module(shared_vocab_config):
    return fastqa_like_with_min_crossentropy_loss_factory(shared_vocab_config, fastqa_model)

def BIGRU(x, hidden_size):
    input_x = tf.transpose(x, [1, 0, 2])
    input_x = tf.unstack(input_x)

	# define the forward and backward lstm cells
    gru_fw_cell = rnn_cell.GRUCell(hidden_size)
    gru_bw_cell = rnn_cell.GRUCell(hidden_size)
    output, _, _ = rnn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, input_x, dtype=tf.float32)

	# output transformation to the original tensor type
    output = tf.stack(output)
    output = tf.transpose(output, [1, 0, 2])
    return output
def _reverse(input_, seq_lengths, seq_dim, batch_dim):
  if seq_lengths is not None:
    return array_ops.reverse_sequence(
        input=input_, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)
  else:
    return array_ops.reverse(input_, axis=[seq_dim])
def run_match_lstm(encoded_rep, masks,size):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks

        match_lstm_cell_attention_fn = lambda curr_input, state : tf.concat([curr_input, state], axis = -1)
        query_depth = encoded_question.get_shape()[-1]


        # output attention is false because we want to output the cell output and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_match_lstm = BahdanauAttention(query_depth, encoded_question, memory_sequence_length = masks_question)
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, state_is_tuple = True)
            lstm_attender  = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False, cell_input_fn = match_lstm_cell_attention_fn)

            # we don't mask the passage because masking the memories will be handled by the pointerNet
            reverse_encoded_passage = _reverse(encoded_passage, masks_passage, 1, 0)

            output_attender_fw, _ = tf.nn.dynamic_rnn(lstm_attender, encoded_passage, dtype=tf.float32, scope ="rnn")    
            output_attender_bw, _ = tf.nn.dynamic_rnn(lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope = "rnn")

            output_attender_bw = _reverse(output_attender_bw, masks_passage, 1, 0)

        
        output_attender = tf.concat([output_attender_fw, output_attender_bw], axis = -1) # (-1, P, 2*H)
        return output_attender
def mask(seqence_length,maxlen):
    batch_size = tf.shape(seqence_length)[0]
    lengths_transposed = tf.expand_dims(seqence_length, 1)
    # Make a 4 x 8 matrix where each row contains [0, 1, ..., 7]
    ranges = tf.range(0,maxlen,1)
    range_row = tf.expand_dims(ranges, 0)
    # Use the logical operations to create a mask
    mask = tf.less(range_row, lengths_transposed)
    # Use the select operation to select between 1 or 0 for each value.
    result = tf.where(mask, tf.ones([batch_size, maxlen]), tf.zeros([batch_size, maxlen]))
    return result   
def fastqa_model(shared_vocab_config, emb_question, question_length,
                 emb_support, support_length,
                 unique_word_chars, unique_word_char_length,
                 question_words2unique, support_words2unique,
                 word_in_question,slot_list,
                 correct_start, answer2question, keep_prob, is_eval):
    
    """
    fast_qa model
    Args:
        shared_vocab_config: has at least a field config (dict) with keys "rep_dim", "rep_dim_input"
        emb_question: [Q, L_q, N]
        question_length: [Q]
        emb_support: [Q, L_s, N]
        support_length: [Q]
        unique_word_chars
        unique_word_char_length
        question_words2unique
        support_words2unique
        word_in_question: [Q, L_s]
        correct_start: [A], only during training, i.e., is_eval=False
        answer2question: [A], only during training, i.e., is_eval=False
        keep_prob: []
        is_eval: []

    Returns:
        start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
    """
    with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
        # Some helpers
        batch_size = tf.shape(question_length)[0]
        max_question_length = tf.reduce_max(question_length)
        support_mask = tfutil.mask_for_lengths(support_length, batch_size)
        question_binary_mask = tfutil.mask_for_lengths(question_length, batch_size, mask_right=False, value=1.0)

        input_size = shared_vocab_config.config["repr_dim_input"]
        size = shared_vocab_config.config["repr_dim"]
        with_char_embeddings = shared_vocab_config.config.get("with_char_embeddings", False)
        
        # set shapes for inputs
        emb_question.set_shape([None, None, input_size])
        emb_support.set_shape([None, None, input_size])
        #slot embedding part
        slot_embeddings = tf.get_variable("slot_embeddings",
                                          [64, size])
        embedded_slot_ids = tf.gather(slot_embeddings, slot_list)
        
        if with_char_embeddings:
            # compute combined embeddings
            [char_emb_question, char_emb_support] = conv_char_embedding_alt(shared_vocab_config.config["char_vocab"],
                                                                            size,
                                                                            unique_word_chars, unique_word_char_length,
                                                                            [question_words2unique,
                                                                             support_words2unique])

            emb_question = tf.concat([emb_question, char_emb_question], 2)
            emb_support = tf.concat([emb_support, char_emb_support], 2)
            input_size += size

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

        # compute encoder features
        question_features = tf.ones(tf.stack([batch_size, max_question_length, 2]))

        v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size],
                                 initializer=tf.constant_initializer(1.0))

        wiq_w = tf.matmul(emb_question * v_wiqw, emb_support, adjoint_b=True)
        wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

        wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

        # [B, L , 2]
        support_features = tf.concat([tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w, 2)], 2)

        # highway layer to allow for interaction between concatenated embeddings
        if with_char_embeddings:
            all_embedded = tf.concat([emb_question, emb_support], 1)
            all_embedded = tf.contrib.layers.fully_connected(all_embedded, size,
                                                             activation_fn=None,
                                                             weights_initializer=None,
                                                             biases_initializer=None,
                                                             scope="embeddings_projection")

            all_embedded_hw = highway_network(all_embedded, 1)

            emb_question = tf.slice(all_embedded_hw, [0, 0, 0], tf.stack([-1, max_question_length, -1]))
            emb_support = tf.slice(all_embedded_hw, tf.stack([0, max_question_length, 0]), [-1, -1, -1])

            emb_question.set_shape([None, None, size])
            emb_support.set_shape([None, None, size])

        # variational dropout
        dropout_shape = tf.unstack(tf.shape(emb_question))
        dropout_shape[1] = 1

        [emb_question, emb_support] = tf.cond(is_eval,
                                              lambda: [emb_question, emb_support],
                                              lambda: fixed_dropout([emb_question, emb_support],
                                                                    keep_prob, dropout_shape))

        # extend embeddings with features
        emb_question_ext = tf.concat([emb_question, question_features], 2)
        emb_support_ext = tf.concat([emb_support, support_features], 2)
        
        # encode question and support
        rnn = tf.contrib.rnn.LSTMBlockFusedCell
        encoded_question = birnn_with_projection(size, rnn, emb_question_ext, question_length,
                                                 projection_scope="question_proj")

        encoded_support = birnn_with_projection(size, rnn, emb_support_ext, support_length,
                                                share_rnn=True, projection_scope="support_proj")
        
       
        start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
            fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                correct_start, answer2question, is_eval,embedded_slot_ids,
                                beam_size=shared_vocab_config.config.get("beam_size", 1))
        
        span = tf.concat([tf.expand_dims(predicted_start_pointer, 1), tf.expand_dims(predicted_end_pointer, 1)], 1)

        return start_scores, end_scores, span


# ANSWER LAYER
def fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                        correct_start, answer2question, is_eval, embedded_slot_ids, beam_size=1):
    beam_size = tf.cond(is_eval, lambda: tf.constant(beam_size, tf.int32), lambda: tf.constant(1, tf.int32))
    batch_size = tf.shape(question_length)[0]
    answer2question = tf.cond(is_eval, lambda: tf.range(0, batch_size, dtype=tf.int32), lambda: answer2question)
    input_size = encoded_support.get_shape()[-1].value
    support_states_flat = tf.reshape(encoded_support, [-1, input_size])
    
    # computing single time attention over question
    attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="question_attention")
    q_mask = tfutil.mask_for_lengths(question_length, batch_size)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])
    
    slot_q_state=tf.concat([tf.expand_dims(question_state,dim=1),tf.expand_dims(embedded_slot_ids,dim=1)],1)
    attention_scores2 = tf.contrib.layers.fully_connected(slot_q_state, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="slot_question_attention")
    slot_question_attention_weights = tf.nn.softmax(attention_scores2, 1, name="slot_question_attention_weights")
    slot_question_state = tf.reduce_sum(slot_question_attention_weights * slot_q_state, [1])
    # Prediction
    # start
    '''
    with tf.variable_scope('coattention'):
            s_encoding = tf.contrib.layers.fully_connected(encoded_support, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="q_start_inter")
            q_encoding = tf.tanh(encoded_question)
            c_encoding = tf.tanh(s_encoding)
            q_variation = tf.transpose(q_encoding, perm=[0, 2, 1])
          # compute affinity matrix, (batch_size, context+1, question+1)
            L = tf.matmul(c_encoding, q_variation)
          # shape = (batch_size, question+1, context+1)
            L_t = tf.transpose(L, perm=[0, 2, 1])
          # normalize with respect to question
            a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
          # normalize with respect to context
            a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
          # summaries with respect to question, (batch_size, question+1, hidden_size)
            c_q = tf.matmul(a_q, c_encoding)
            c_q_emb = tf.concat([q_variation, tf.transpose(c_q, perm=[0, 2 ,1])],1)
          # summaries of previous attention with respect to context
            c_d = tf.matmul(c_q_emb, a_c, adjoint_b=True)
          # final coattention context, (batch_size, context+1, 3*hidden_size)
            co_att = tf.concat([c_encoding, tf.transpose(c_d, perm=[0, 2, 1])],2)
    
    with tf.variable_scope('encoder'):
          # LSTM for coattention encoding
          cell_fw = tf.nn.rnn_cell.LSTMCell(size/10)
          cell_bw = tf.nn.rnn_cell.LSTMCell(size/10)
          # compute coattention encoding
          u, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw, cell_bw, co_att,
              sequence_length=support_length,
              dtype=tf.float32)
          output_match = tf.concat(u,2)
    '''
    max_support_length= tf.reduce_max(support_length)
    max_question_length= tf.reduce_max(question_length)
    question_mask=mask(question_length,maxlen=max_question_length)
    support_mask=mask(support_length,maxlen=max_support_length)
    question_mask=tf.cast(question_mask,tf.float32)
    support_mask=tf.cast(support_mask,tf.float32)
    output_match=match_utils.bilateral_match_func2(encoded_question, encoded_support,
                    question_length, support_length, question_mask, support_mask, MP_dim=30, input_dim=size, 
                    with_filter_layer=True, context_layer_num=1, context_lstm_dim=size/2,is_training=True,dropout_rate=0.5,
                    with_match_highway=True,aggregation_layer_num=1, aggregation_lstm_dim=size/2,highway_layer_num=1,
                    with_aggregation_highway=False,with_lex_decomposition=False,lex_decompsition_dim=300,
                    with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                    with_left_match=True, with_right_match=True, with_mean_aggregation=True)
    '''
    output_match=run_match_lstm([encoded_question,encoded_support],[question_length,support_length],size)
    '''
    start_input = tf.concat([tf.expand_dims(slot_question_state, 1) * encoded_support,
                             encoded_support,output_match], 2)

    q_start_inter = tf.contrib.layers.fully_connected(slot_question_state, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="q_start_inter")

    q_start_state = tf.contrib.layers.fully_connected(start_input, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      biases_initializer=None,
                                                      scope="q_start") + tf.expand_dims(q_start_inter, 1)

    start_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_start_state), 1,
                                                     activation_fn=None,
                                                     weights_initializer=None,
                                                     biases_initializer=None,
                                                     scope="start_scores")
    start_scores = tf.squeeze(start_scores, [2])

    support_mask = tfutil.mask_for_lengths(support_length, batch_size)
    start_scores = start_scores + support_mask

    # probs are needed during beam search
    start_probs = tf.nn.softmax(start_scores)

    predicted_start_probs, predicted_start_pointer = tf.nn.top_k(start_probs, beam_size)

    # use correct start during training, because p(end|start) should be optimized
    predicted_start_pointer = tf.gather(predicted_start_pointer, answer2question)
    predicted_start_probs = tf.gather(predicted_start_probs, answer2question)

    start_pointer = tf.cond(is_eval, lambda: predicted_start_pointer, lambda: tf.expand_dims(correct_start, 1))

    # flatten again
    start_pointer = tf.reshape(start_pointer, [-1])
    answer2questionwithbeam = tf.reshape(tf.tile(tf.expand_dims(answer2question, 1), tf.stack([1, beam_size])), [-1])

    offsets = tf.cast(tf.range(0, batch_size) * tf.reduce_max(support_length), dtype=tf.int32)
    offsets = tf.gather(offsets, answer2questionwithbeam)
    u_s = tf.gather(support_states_flat, start_pointer + offsets)

    start_scores = tf.gather(start_scores, answer2questionwithbeam)
    start_input = tf.gather(start_input, answer2questionwithbeam)
    encoded_support = tf.gather(encoded_support, answer2questionwithbeam)
    slot_question_state = tf.gather(slot_question_state, answer2questionwithbeam)
    support_mask = tf.gather(support_mask, answer2questionwithbeam)

    # end
    end_input = tf.concat([tf.expand_dims(u_s, 1) * encoded_support, start_input], 2)

    q_end_inter = tf.contrib.layers.fully_connected(tf.concat([slot_question_state, u_s], 1), size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    scope="q_end_inter")

    q_end_state = tf.contrib.layers.fully_connected(end_input, size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    biases_initializer=None,
                                                    scope="q_end") + tf.expand_dims(q_end_inter, 1)

    end_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_end_state), 1,
                                                   activation_fn=None,
                                                   weights_initializer=None,
                                                   biases_initializer=None,
                                                   scope="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask

    def mask_with_start(scores):
        return scores + tfutil.mask_for_lengths(tf.cast(start_pointer, tf.int32),
                                                batch_size * beam_size, tf.reduce_max(support_length),
                                                mask_right=False)

    end_scores = tf.cond(is_eval, lambda: mask_with_start(end_scores), lambda: end_scores)

    # probs are needed during beam search
    end_probs = tf.nn.softmax(end_scores)
    predicted_end_probs, predicted_end_pointer = tf.nn.top_k(end_probs, 1)
    predicted_end_probs = tf.reshape(predicted_end_probs, tf.stack([-1, beam_size]))
    predicted_end_pointer = tf.reshape(predicted_end_pointer, tf.stack([-1, beam_size]))

    predicted_idx = tf.cast(tf.argmax(predicted_start_probs * predicted_end_probs, 1), tf.int32)
    predicted_idx = tf.stack([tf.range(0, tf.shape(answer2question)[0], dtype=tf.int32), predicted_idx], 1)

    predicted_start_pointer = tf.gather_nd(predicted_start_pointer, predicted_idx)
    predicted_end_pointer = tf.gather_nd(predicted_end_pointer, predicted_idx)

    return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer
