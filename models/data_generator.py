import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    :return: data generator object
    """
    def __init__(self, data_list, index_dict, list_IDs, labels, relations_list, batch_size=32, shuffle=True):
        # Initialization
        # self.reader = reader
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.relations = relations_list
        self.data_num = 0
        self.data_list = data_list
        self.index_dict = index_dict

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: int
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the current training item
        :return: tuple
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, y = self.__index_reader(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples'
        :param list_IDs_temp: the list of IDs of the target batch
        :return: tuple
        """
        # Initialization
        y = []
        v_q_words = []
        v_d_words = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            q_words = self.reader.get_query(self.relations[ID][0])
            v_q_words.append(q_words)
            d_words = self.reader.get_document(self.relations[ID][1])
            v_d_words.append(d_words)
            # Store class
            y.append(self.labels[ID])

        X = [np.array(v_q_words), np.array(v_d_words)]

        return X, np.array(y)

    def __index_reader(self, list_IDs_temp):
        """
        Reads the index dictionary and content list of documents
        :param list_IDs_temp:
        :return: tuple
        """
        # Initialization
        y = []
        v_q_words = []
        v_d_words = []

        # Read data
        for i, ID in enumerate(list_IDs_temp):
            q_words = self.data_list[self.index_dict[self.relations[ID][0]]]
            v_q_words.append(q_words)
            d_words = self.data_list[self.index_dict[self.relations[ID][1]]]
            v_d_words.append(d_words)
            y.append(self.labels[ID])

        X = [np.array(v_q_words), np.array(v_d_words)]

        return X, np.array(y)
