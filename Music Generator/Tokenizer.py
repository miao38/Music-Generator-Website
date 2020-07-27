'''
tokenizer class
this class will creating a mapping for the sequence
'''
class Tokenizer:
    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.num_word = 0
        self.unique_word = 0
        self.note_freq = {}

    '''transform a list of notes from strings to indexes
        list_array is a list of notes in string format'''
    def transform(self, list_array):
        transformed = []
        for i in list_array:
            transformed.append([self.notes_to_index[note] for note in i])
        return np.array(transformed, dtype = np.int32)

    '''partial fir on the dictionary of the tokenizer
        notes is a list of notes'''
    def partial_fit(self, notes):
        for note in notes:
            note_str = ",".join(str(n) for n in note)
            if note_str in self.note_freq:
                self.note_freq[note_str] += 1
                self.num_word += 1
            else:
                self.note_freq[note_str] = 1
                self.unique_word += 1
                self.num_word += 1
                self.notes_to_index[note_str] =self.unique_word
                self.index_to_notes[self.unique_word] = note_str

    '''add a new note to the dictionary
        note is the new note to be added as a string'''
    def add_new_note(self, note):
        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note] = self.unique_word
        self.index_to_notes[self.unique_word] = note