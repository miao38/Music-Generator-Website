import pretty_midi
import numpy as np
from numpy.random import choice
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from SeqSelfAttention import SeqSelfAttention
import random
import os
import glob
import pickle
from Tokenizer import *

def starter(training):
    from Tokenizer import Tokenizer
    model = tf.keras.models.load_model("model_25epochs.h5", custom_objects=SeqSelfAttention.get_custom_objects())
    tokenizer = pickle.load(open("tokenizer25.p", "rb"))
    #generate from random
    max_generate = 200
    unique_notes = tokenizer.unique_word
    seq_len = 200
    generate = generate_from_random(unique_notes, seq_len)
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    return write_midi_file(generate, tokenizer, "song #1.mid", start=seq_len - 1, fs=7, max_generate=max_generate)
'''
    #generate from a note
    max_generate = 300
    unique_notes = tokenizer.unique_word  # same as above
    seq_len = 300
    generate = generate_from_one_note(tokenizer, "72")
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    write_midi_file(generate, tokenizer, "one note test.mid", start=seq_len - 1, fs=8, max_generate=max_generate)
'''

'''
converts a piano roll array into a PrettyMidi object with a single instrument
'''
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

'''
getting a list of all midi files in the folder
'''
def get_midi(folder="Songs/*.mid", s=5):
    list_all_midi = glob.glob(folder)
    random.seed(s)
    random.shuffle(list_all_midi)
    return list_all_midi

'''
generates the input and the target of our deep learning for one song
passed in dictionary will have start and end times (start, end)
"e" means that there are no notes played in that time
first note we append "e" to the list 49 times and set the start time to the first time step in the dictionary
second note we append "e" to the list 48 times and so on
'''
def generate_input_and_target(dict_keys_times, seq_len = 50):
    #Getting the start and end time
    start = list(dict_keys_times.keys())[0]
    end = list(dict_keys_times.keys())[-1]
    train = []
    target = []
    for i, time in enumerate(range(start, end)):
        appended_train = []
        appended_target = []
        iterate = 0
        flag_target = False #flag to append the test list
        if i < seq_len:
            iterate = seq_len - i - 1
            for j in range(iterate): #adding "e" to the seq list
                appended_train.append("e")
                flag_target = True

        for j in range(iterate, seq_len):
            index = time - (seq_len - j - 1)
            if index in dict_keys_times:
                appended_train.append(",".join(str(x) for x in dict_keys_times[index]))
            else:
                appended_train.append("e")

        #add time + 1 to appended_target
        if time + 1 in dict_keys_times:
            appended_target.append(",".join(str(x) for x in dict_keys_times[time + 1]))
        else:
            appended_target.append("e")
        train.append(appended_train)
        target.append(appended_target)
    return train, target


'''
creating batch music that will be used to be input and output of the neural network
fs is the sampling freq of the columns (each column is spaced apart by 1 fs second)
'''
def generate_batch_song(list_all_midi, batch_music = 16, start = 0, fs = 30, seq_len = 50, use_tqdm = False):
    assert len(list_all_midi) >= batch_music
    dict_time_notes = generate_dict_time_notes(list_all_midi, batch_music, start, fs, use_tqdm=use_tqdm)
    list_music = process_notes_in_song(dict_time_notes, seq_len)
    collected_input = []
    collected_target = []
    for music in list_music:
        train, target = generate_input_and_target(music, seq_len)
        collected_input += train
        collected_target += target
    return collected_input, collected_target

'''
generates a dictionary of music to piano roll
'''
def generate_dict_time_notes(list_all_midi, batch_song = 25, start=0, fs=30, use_tqdm=True): #change batch_song back to 16 when using 5 epochs
    assert len(list_all_midi) >= batch_song
    dict_time_notes = {}
    process_midi = range(start, min(start + batch_song, len(list_all_midi))) if use_tqdm else range(start,  min(start + batch_song, len(list_all_midi)))
    for i in process_midi:
        midi_file_name = list_all_midi[i]
        if use_tqdm:
            process_midi.set_description("Processing {}".format(midi_file_name))
        try: #handle exceptions on malformated MIDI files
            midi_format = pretty_midi.PrettyMIDI(midi_file_name)
            piano_midi = midi_format.instruments[0] #gets the piano ones
            piano_roll = piano_midi.get_piano_roll(fs=fs)
            dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("Broken File: {}".format(midi_file_name))
            pass
    return dict_time_notes

'''
generates input and target of the neural network for one song
'''
def generate_input_and_target(dict_keys_time, seq_len = 50):
    #getting the start and end time
    start = list(dict_keys_time.keys())[0]
    end = list(dict_keys_time.keys())[-1]
    list_train = []
    list_target = []
    for index, time in enumerate(range(start, end)):
        list_append_train = []
        list_append_target = []
        i = 0
        flag = False #flag to append the test list
        if index < seq_len:
            i = seq_len - index - 1
            for j in range(i): #add "e" to the seq list
                list_append_train.append("e")
                flag = True
        for j in range(i, seq_len):
            index = time - (seq_len - j - 1)
            if index in dict_keys_time:
                list_append_train.append(",".join(str(x) for x in dict_keys_time[index]))
            else:
                list_append_train.append("e")
        #add time + 1 to the list_append_target
        if time + 1 in dict_keys_time:
            list_append_target.append(",".join(str(x) for x in dict_keys_time[time + 1]))
        else:
            list_append_target.append("e")
        list_train.append(list_append_train)
        list_target.append(list_append_target)
    return list_train, list_target

'''
iterate the dictionary of piano rolls into dictionary of timesteps and notes played
'''
def process_notes_in_song(dict_time_notes, seq_len = 50):
    list_of_dict_keys_times = []
    for key in dict_time_notes:
        sample = dict_time_notes[key]
        times = np.unique(np.where(sample > 0)[1])
        index = np.where(sample > 0)
        dict_keys_time = {}
        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes
        list_of_dict_keys_times.append(dict_keys_time)
    return list_of_dict_keys_times

'''
creating the model
'''
def create_model(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=128, dense_unit=64):
    inputs = tf.keras.layers.Input(shape=(seq_len))
    embedding = tf.keras.layers.Embedding(input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    forward_pass, att_vector = SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=tf.keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4)(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass, att_vector2 = SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=tf.keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4)(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_unit))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
    outputs = tf.keras.layers.Dense(unique_notes + 1, activation="softmax")(forward_pass)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_rnn')
    return model

class TrainModel:
    def __init__(self, epochs, note_tokenizer, sampled_200_midi, fps, batch_nnet_size,
                 batch_song, optimizer, checkpoint, loss_fn, checkpoint_prefix,
                 total_songs, seq_len, model):
        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.sampled_200_midi = sampled_200_midi
        self.fps = fps
        self.batch_nnet_size = batch_nnet_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.loss_fn = loss_fn
        self.checkpoint_prefix = checkpoint_prefix
        self.total_songs = total_songs
        self.seq_len = seq_len
        self.model = model

    def train(self):
        for epoch in range(self.epochs):
            #for each epochs, we shuffe the list of all the datasets
            random.shuffle(self.sampled_200_midi)
            loss_total = 0
            steps = 0
            steps_nnet = 0
            #iterate through all songs by self.song_size
            for i in range(0, self.total_songs, self.batch_song):
                steps += 1
                inputs_nnet_large, outputs_nnet_large = generate_batch_song(
                    self.sampled_200_midi, self.batch_song, start=i, fs=self.fps,
                    seq_len=self.seq_len, use_tqdm=False)
                inputs_nnet_large = np.array(self.note_tokenizer.transform(inputs_nnet_large), dtype=np.int32)
                outputs_nnet_large = np.array(self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)
                index_shuffled = np.arange(start=0, stop=len(inputs_nnet_large))
                np.random.shuffle(index_shuffled)
                for nnet_steps in range(0, len(index_shuffled)):
                    steps_nnet += 1
                    current_index = index_shuffled[nnet_steps:nnet_steps+self.batch_nnet_size]
                    inputs_nnet = inputs_nnet_large[current_index]
                    outputs_nnet = outputs_nnet_large[current_index]
                    #make sure no exception is thrown by tensorflow on autograph
                    if len(inputs_nnet) // self.batch_nnet_size != 1:
                        break
                    loss = self.train_step(inputs_nnet, outputs_nnet)
                    loss_total += tf.math.reduce_sum(loss)
                    if steps_nnet % 20 == 0:
                        print("epochs {} | Steps {} | total loss: {}".format(epoch + 1, steps_nnet, loss_total))
            checkpoint.save(file_prefix = self.checkpoint_prefix)

    @tf.function
    def train_step(self, inputs, target):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(target, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

'''
generate 50 random notes as the start of the music
'''
def generate_from_random(unique_notes, seq_len=50):
    generate = np.random.randint(0, unique_notes, seq_len).tolist()
    return generate

'''
use 49 empty notes followed by a start note of our choice
'''
def generate_from_one_note(note_tokenizer, new_note="35"):
    generate = [note_tokenizer.notes_to_index["e"] for i in range(49)]
    generate += [note_tokenizer.notes_to_index[new_note]]
    return generate

def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len= 50):
    for i in range(max_generated):
        test_input = np.array([generate])[:, i:i+seq_len]
        predicted_note = model.predict(test_input)
        random_note_pred = choice(unique_notes + 1, 1, replace=False, p=predicted_note[0])
        generate.append(random_note_pred[0])
    return generate

def write_midi_file(generate, tokenizer, midi_file_name = "result.mid", start=49, fs=8, max_generate=1000):
    note_string = [tokenizer.index_to_notes[i] for i in generate]
    array_piano_roll = np.zeros((128, max_generate+1), dtype=np.int16)
    for i, note in enumerate(note_string[start:]):
        if note == "e":
            pass
        else:
            split_note = note.split(",")
            for j in split_note:
                array_piano_roll[int(j), i] = 1
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    #generate_to_midi.write(midi_file_name)
    new_file = open(midi_file_name, 'wb')
    generate_to_midi.write(new_file)
    new_file.close()
    new_file = open(midi_file_name, 'rb')
    return(new_file)

if(__name__ == '__main__'):
    starter("no")