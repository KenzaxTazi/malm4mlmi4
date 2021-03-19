import abc

import numpy as np
import torch

from utils import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

__all__ = ['SinusoidGenerator']

def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)

def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')

class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args: 
        generators (function): Function that generates an element
        num_elements (int): Number of elements to generate
    """   

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0
    
    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

        def __iter__(self):
            return self

class DataGenerator(metaclass=abc.ABCMeta):
    """Data genetator for sinusoid samples.
    
    Args:
        batch_size(int, optional): Batch size. Defaults to 16.
        num_tasks(int, optional): Number of tasks to generator per epoch. 
            Defaults to 256
        x_range (tuple[float], optional): Range of the inputs. 
        Defaults to [-2,2].
        
        [TO DO: complete explanation] 
    
    """

    def __init__(self,
                 batch_size = 16,
                 num_tasks = 256,
                 x_range = (-5, 5),
                 A_range = (0.1, 5),
                 P_range = (0, np.pi),
                 min_support_points=10,
                 max_support_points=10,
                 min_query_points=10,
                 max_query_points=10,
                 permute_inds=True,
                 ):

        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.A_range = A_range
        self.P_range = P_range
        self.max_support_points = max_support_points
        self.max_query_points = max_query_points
        self.min_support_points = min_support_points
        self.min_query_points = min_query_points
        self.permute_inds = permute_inds

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs 'x'.

        Args:
            x(vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs 'x'.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys 'x', 'y', 'x_support',
                'y_support', 'x_query', and 'y_query'.
        """
        task = {'x': [],
                'y': [],
                'x_support': [],
                'y_support': [],
                'x_query': [],
                'y_query': [],
                'amplitudes': [],
                'phases': []
                }
        
        #Determine number of context (train) and query (test) points
        num_train_points = np.random.randint(self.min_support_points, self.max_support_points + 1)
        num_test_points = np.random.randint(self.min_query_points, self.max_query_points + 1)
        num_points = num_train_points + num_test_points
        
        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = np.sort(_rand(self.x_range, num_points))
            A = _rand(self.A_range, 1)
            phase = _rand(self.P_range, 1)
            y = np.squeeze(np.sin(x + phase) * A)
            #y = self.sample(x)

            # Determine indices for train and test set.
            
            if self.permute_inds:
                inds = np.random.permutation(x.shape[0])
            else:
                x = np.sort(x)
                inds = np.arange(x.shape[0])

            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['y'].append(y[np.argsort(x)])
            task['x_support'].append(x[inds_train])
            task['y_support'].append(y[inds_train])
            task['x_query'].append(x[inds_test])
            task['y_query'].append(y[inds_test])
            task['amplitudes'].append(A)
            task['phases'].append(phase)

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(device)
                for k, v in task.items()}
        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)

class SinusoidGenerator(DataGenerator):
    """Genrate samples from sinusoid functions.

    Further takes in keyword arguments for :class:'.data.DataGenerator'.

    Args: 

        [TO DO]

    """

    def __init__(self, **kw_args):
        DataGenerator.__init__(self,**kw_args)

    def sample(self, x):
        A = _rand(self.A_range, 1)
        phase = _rand(self.P_range, 1)
        return np.squeeze (np.sin(x + phase) * A)


if __name__ == '__main__':
    print("Executed")
    SinusoidGenerator(permute_inds = False).generate_task()
    print("Executed")
