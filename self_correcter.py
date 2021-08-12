import numpy as np
import operator

class Correcter(object):
    def __init__(self, size_of_data, num_of_classes, queue_size):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.queue_size = queue_size

        self.accumulator = {}
        self.prec = np.zeros(self.size_of_data, dtype=int)

        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            self.all_predictions[i] = np.zeros(queue_size, dtype=int)

        # self.softmax_record = {}
        # for i in range(size_of_data):
        #     self.softmax_record[i] = []

        self.certainty_array = {}
        for i in range(size_of_data):
            self.certainty_array[i] = 1

        self.p_dict = np.zeros(self.num_of_classes, dtype=float)
        self.max_certainty = -np.log(1.0/float(self.queue_size))
        self.clean_key = []

        self.update_counters = np.zeros(size_of_data, dtype=int)

    def async_update_prediction_matrix(self, ids, softmax_matrix):#更新 得到最近十次的预测
        for i in range(len(ids)):
            id = ids[i].item()
            # predicted_label = np.argmax(softmax_matrix[i])
            cur_index = self.update_counters[id] % self.queue_size
            self.all_predictions[id][cur_index] = softmax_matrix[i]
            #self.softmax_record[id] = epoch_loss[i]

            self.update_counters[id] += 1

    def get_certainty_array(self, ids):   #Fxi
        # accumulator = {}
        for i in range(len(ids)):
            id = ids[i].item()

            predictions = self.all_predictions[id]
            self.accumulator.clear()
            self.p_dict = np.zeros(self.num_of_classes, dtype=float)

            for prediction in predictions:
                if prediction not in self.accumulator:
                    self.accumulator[prediction] = 1
                else:
                    self.accumulator[prediction] = self.accumulator[prediction] + 1

            self.prec[id] = max(self.accumulator, key=self.accumulator.get)  #数组

            # p_dict = np.zeros(self.num_of_classes, dtype=float)
            for key, value in self.accumulator.items():
                self.p_dict[key] = float(value) / float(self.queue_size)

            # based on entropy
            negative_entropy = 0.0
            for i in range(len(self.p_dict)):
                if self.p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += self.p_dict[i] * np.log(self.p_dict[i])
            certainty = - negative_entropy / self.max_certainty
            self.certainty_array[id] = certainty

    def separate_clean_and_unclean_keys(self,noise_rate,id_hard):  #区分分布内噪声与分布外噪声样本
        loss_map = {}
        self.clean_key = []    #list
        num_clean_instances = int(np.ceil(float(len(id_hard)) * (1.0 - noise_rate)))
        for i in range(len(id_hard)):
            id = id_hard[i]
            loss_map[i] = self.certainty_array[id]

        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))  #升序

        index = 0
        for value in loss_map.values():
            if index  ==  num_clean_instances:
                return value
            index += 1
        loss_map.clear()


    # def separate_clean_and_unclean_samples(self, ids, images, labels):
    #     clean_batch = MiniBatch()
    #     unclean_batch = MiniBatch()
    #
    #     for i in range(len(ids)):
    #         if ids[i] in self.unclean_key:
    #             unclean_batch.append(ids[i], images[i], labels[i])
    #         else:
    #             clean_batch.append(ids[i], images[i], labels[i])
    #
    #     return clean_batch, unclean_batch
    #
    #
    #
    # # def get_certainty_label(self, labels, ids, softmax_matrix):
    # #     for i in range(len(ids)):
    # #         id = ids[i].item()
    # #         predicted_label = np.argmax(softmax_matrix[i])
    # #         f_x = self.certainty_array[id]
    # #         if f_x > 0.5:
    # #             labels[id] = predicted_label
    # #
    # #     return labels
    #
    # def patch_clean_with_corrected_sample_batch(self, ids, images, labels):
    #     # 1. update certainly array
    #     self.get_certainty_array(ids)
    #     # 2. separate clean and unclean samples
    #     clean_batch, unclean_batch = self.separate_clean_and_unclean_samples(ids, images, labels)
    #     return clean_batch.ids
    #
    # #     for i in range(len(clean_batch.ids)):
    # #         id = clean_batch.ids[i].item()
    # #         cur_index = self.update_counters[id] % self.queue_size
    # #         predicted_label = self.all_predictions[id][cur_index]
    # #         labels_final = clean_batch.labels
    # #         f_x = self.certainty_array[id]
    # #         if f_x > 0.5:
    # #             clean_batch.labels[id] = predicted_label
    # #
    # #     return clean_batch.ids, clean_batch.images, clean_batch.labels
    #
    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.queue_size, dtype=int)
