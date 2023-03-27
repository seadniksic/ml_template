import numpy as np


class SupervisedMLFramework:

    def __init__(self, model, data, labels, split, k: int, autotune: bool, is_custom_model: bool) -> None:
        self.model = model
        self.data = data
        self.labels = labels
        self.autotune = autotune
        self.train_proportion = split[0]
        self.validation_proportion = split[1]
        self.test_proportion = split[2]
        self.data_len = len(labels)
        self.is_custom_model = is_custom_model
        self.k = k

    def preprocess(self, preprocessing_function) -> None:
        self.data = preprocessing_function(self.data)

    def train(self):
        if self.autotune:
            #Do sklearn gridsearchcv here.  Will need the model to implement the sklearn estimator interface which im not sure about
            pass
        else:
            #Perform k fold cross validation on the model.  Assume default params are already chosen

            #Separate data into train, test

                #Generate indices in random order (no repeats)

                indices = np.random.permutation(range(self.data_len))
                train_length = int((self.train_proportion + self.validation_proportion)  * self.data_len)
                train_indices = indices[: train_length]
                test_indices = indices[train_length:]
                

                #Take indices and create train / test data and labels

                train_data = self.data[train_indices]
                train_labels = self.labels[train_indices]
                test_data = self.data[test_indices]
                test_labels = self.labels[test_indices]

                #Do k fold cross validation on train portion.  Make sure to shuffle the train set each time.
                fold_size = int(len(train_data) / self.k)
                for split in range(1, self.k + 1):
                    #If length of data isn't perfectly divisible by k, give the last fold whatever's left over (a max of k-1 samples) 
                    k_fold_indices = list(range(train_data))
                    validation_data = k_fold_indices[(split -1) *fold_size: split *fold_size] if split != self.k else train_data[(split -1) *fold_size:]
                    

                if not self.is_custom_model:
                    self.model.fit(train_data, train_labels)
                else:
                    #TODO
                    pass



            #Separate train data into 

    #TODO
    def test():
        pass
    