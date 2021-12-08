from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Loss(Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

        self._agg_loss = 0
        self._dis_loss = 0
        self._text_loss = 0
        self._kernel_loss = 0

    def update(self, output):
        avg_loss, agg_loss, dis_loss, text_loss, kernel_loss = self._loss_fn(*output)

        if len(avg_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = output[1].shape[0]
        self._num_examples += N
        self._sum += avg_loss.item() * N

        self._agg_loss += agg_loss.item() * N
        self._dis_loss += dis_loss.item() * N
        self._text_loss += text_loss.item() * N
        self._kernel_loss += kernel_loss.item() * N


    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')

        print(f'aggregation loss: {self._agg_loss / self._num_examples}')
        print(f'distance loss: {self._dis_loss / self._num_examples}')
        print(f'text region loss: {self._text_loss / self._num_examples}')
        print(f'kernel loss: {self._kernel_loss / self._num_examples}')

        return self._sum / self._num_examples
