""" Code for loading data. """
import numpy as np
import os
import random


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, datasource, num_classes=1, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes  # by default 1 (only relevant for classification problems)

        if datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1

        elif datasource == 'sinusoid_linear':
            # mix of sinusoid and linear functions.
            self.generate = self.generate_sinusoid_linear_batch
            # for sine
            self.amp_range = config.get('amp_range', [1.0, 5.0])  # TODO - this is different from MAML paper.
            self.phase_range = config.get('phase_range', [0, np.pi])
            # for linear
            self.slope_range = config.get('slope_range', [-3.0, 3.0])
            self.bias_range = config.get('bias_range', [-3.0, 3.0])
            # for both
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif datasource == '2dclass_nonlinear':
            self.generate = self.generate_2d_classification
            self.circle = False
            self.nonlinear_boundary = True
            self.amp_range = [0.1, 5.0]
            self.phase_range = [-np.pi, np.pi]
            self.feat_range = [-5.0, 5.0]
            self.dim_input = 2
            self.dim_output = 1
        elif datasource == '2dclass_circle':
            self.generate = self.generate_2d_circle_classification
            self.circle = True
            self.nonlinear_boundary = True
            self.diameter_range = [0.1,2.0]
            self.center_range = [1.0, 4.0]
            self.feat_range = [0.0, 5.0]
            self.dim_input = 2
            self.dim_output = 1

        elif datasource == '2dclass': # 2d binary classification
            self.generate = self.generate_2d_classification
            self.nonlinear_boundary = False
            # decision boundary parameters
            self.slope_range = [-1.0,1.0]
            self.bias_range = [-1.0, 1.0]

            self.feat_range = [-2.0, 2.0]

            self.dim_input = 2
            self.dim_output = 1

    def _reshuffle_tasks(self):
        random.shuffle(self.train_keys)

    def sample_batch(self, attr_idx, data, size=None):
        sampled_attr_idx = np.random.choice(attr_idx,
                                            size=self.num_samples_per_class if size is None else size,
                                            replace=False)
        return data[sampled_attr_idx], sampled_attr_idx

    def generate_2d_classification(self, train=True, input_idx=None):
        if self.nonlinear_boundary:
            slope = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
            bias = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        else:
            slope = np.random.uniform(self.slope_range[0], self.slope_range[1], [self.batch_size])
            bias = np.random.uniform(self.bias_range[0], self.bias_range[1], [self.batch_size])
        inputs = np.random.uniform(self.feat_range[0], self.feat_range[1], [self.batch_size, self.num_samples_per_class, self.dim_input])
        if input_idx is not None:
            lingrid = np.linspace(self.feat_range[0], self.feat_range[1],
                                  np.sqrt(self.num_samples_per_class - input_idx))
            x_grid, y_grid = np.meshgrid(lingrid, lingrid)
            x_grid, y_grid = np.reshape(x_grid, [-1]), np.reshape(y_grid, [-1])
            inputs[:,input_idx:, 0] = x_grid
            inputs[:,input_idx:, 1] = y_grid
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output], dtype=np.int32)

        for task in range(self.batch_size):
            if self.nonlinear_boundary:
                inputs[task,:,1] = np.random.uniform(-2,2, [self.num_samples_per_class]) + slope[task] *np.sin(inputs[task,:,0] - bias[task])
            if np.random.uniform(0,1) > 0.5:
                # positive on top.
                if self.nonlinear_boundary:
                    outputs[task,:,0] = slope[task] *np.sin(inputs[task,:,0] - bias[task]) > inputs[task,:,1]
                else:
                    outputs[task,:,0] = (inputs[task,:,0] * slope[task] + bias[task]) > inputs[task,:,1]
            else:
                if self.nonlinear_boundary:
                    outputs[task,:,0] = slope[task] *np.sin(inputs[task,:,0] - bias[task]) < inputs[task,:,1]
                else:
                    outputs[task,:,0] = (inputs[task,:,0] * slope[task] + bias[task]) < inputs[task,:,1]
        return inputs, outputs, slope, bias

    # def generate_2d_circle_classification(self, train=True, input_idx=None):
    #     centers = np.random.uniform(self.center_range[0], self.center_range[1], [self.batch_size,2])
    #     diameters = np.random.uniform(self.diameter_range[0], self.diameter_range[1], [self.batch_size])
    #     inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
    #     outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output], dtype=np.int32)
    #     num_inner_obj = self.update_batch_size # all positive
    #     num_outer_obj = self.num_samples_per_class - num_inner_obj # both neg and pos
    #     inputs[:,num_inner_obj:, :] = np.random.uniform(self.feat_range[0], self.feat_range[1], [self.batch_size, num_outer_obj, self.dim_input])
    #     for task in range(self.batch_size):
    #         center, diameter = centers[task], diameters[task]
    #         for i in range(num_inner_obj):
    #             r = np.random.uniform(0, diameter/2)
    #             theta = np.random.uniform(0,2*np.pi)
    #             inputs[task,i,0] = r*np.cos(theta) + center[0]
    #             inputs[task,i,1] = r*np.sin(theta) + center[1]
    #         outputs[task, :num_inner_obj, :] = 1
    #         for i in range(num_outer_obj):
    #             idx = num_inner_obj + i
    #             outputs[task,idx,0] = np.sum((inputs[task,idx,:] - center)**2) < (diameter/2.0)**2

        #if input_idx is not None:
        #    lingrid = np.linspace(self.feat_range[0], self.feat_range[1],
        #                          np.sqrt(self.num_samples_per_class - input_idx))
        #    x_grid, y_grid = np.meshgrid(lingrid, lingrid)
        #    x_grid, y_grid = np.reshape(x_grid, [-1]), np.reshape(y_grid, [-1])
        #    inputs[:,input_idx:, 0] = x_grid
        #    inputs[:,input_idx:, 1] = y_grid
        return inputs, outputs, centers, diameters

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase

    def generate_sinusoid_linear_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        slope = np.random.uniform(self.slope_range[0], self.slope_range[1], [self.batch_size])
        bias = np.random.uniform(self.bias_range[0], self.bias_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            if np.random.uniform(0, 1) > 0.5:
                outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
            else:
                outputs[func] = slope[func] * init_inputs[func]+bias[func]
                amp[func] = slope[func] + 100
                phase[func] = bias[func] + 100
        return init_inputs, outputs, amp, phase
