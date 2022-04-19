import torch


input_size = 10
hidden_size = 20
n_layer = 2
seq_len = 5
batch_size = 3
n_direction = 2

gru = torch.nn.GRU(input_size, hidden_size, n_layer, bidirectional=n_direction == 2)
x = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(n_direction*n_layer, batch_size, hidden_size)
o, h = gru(x, h0)
print(o.shape)
print(h.shape)
assert o.shape == (seq_len, batch_size, n_direction*hidden_size)
assert h.shape == (n_direction*n_layer, batch_size, hidden_size)
