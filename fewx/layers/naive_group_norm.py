import torch
from torch.nn import Module, Parameter
from torch.nn import init


class NaiveGroupNorm(Module):
    r"""NaiveGroupNorm implements Group Normalization with the high-level matrix operations in PyTorch.
    It is a temporary solution to export GN by ONNX before the official GN can be exported by ONNX.
    The usage of NaiveGroupNorm is exactly the same as the official :class:`torch.nn.GroupNorm`.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
    Examples::
        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = NaiveGroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = NaiveGroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = NaiveGroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        """
        Initialize the gradients.

        Args:
            self: (todo): write your description
            num_groups: (int): write your description
            num_channels: (int): write your description
            eps: (float): write your description
            affine: (array): write your description
        """
        super(NaiveGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset hyperparameters.

        Args:
            self: (todo): write your description
        """
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        N, C, H, W = input.size()
        assert C % self.num_groups == 0
        input = input.reshape(N, self.num_groups, -1)
        mean = input.mean(dim=-1, keepdim=True)
        var = (input ** 2).mean(dim=-1, keepdim=True) - mean ** 2
        std = torch.sqrt(var + self.eps)

        input = (input - mean) / std
        input = input.reshape(N, C, H, W)
        if self.affine:
            input = input * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
        return input

    def extra_repr(self):
        """
        Return a string representation of - serialization.

        Args:
            self: (todo): write your description
        """
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
