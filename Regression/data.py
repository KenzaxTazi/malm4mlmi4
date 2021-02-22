import abc

import numpy as np
import torch

#from .utils import device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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
                 max_train_points=50,
                 max_test_points=50):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.A_range = A_range
        self.P_range = P_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

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
            dict: A task, which is a dictionary with keys 'x', 'y', 'x_context',
                'y_context', 'x_target', and 'y_target'.
        """
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': []}
        
        #Determine number of test and train points
        num_train_points = np.random.randint(3, self.max_train_points + 1)
        num_test_points = np.random.randint(3, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = _rand(self.x_range, num_points)
            y = self.sample(x)

            # Determine indices for train and test set.
            inds = np.random.permutation(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['y'].append(y[np.argsort(x)])
            task['x_context'].append(x[inds_train])
            task['y_context'].append(y[inds_train])
            task['x_target'].append(x[inds_test])
            task['y_target'].append(y[inds_test])

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
        self.A = _rand(self.A_range, 1)
        self.phase = _rand(self.P_range, 1)
        return np.squeeze (np.sin(x + self.phase) * self.A)
