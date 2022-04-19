import torch


input_size = 10
hidden_size = 20
n_layer = 2
n_direction = 2
seq_len = 5
batch_size = 3

lstm = torch.nn.LSTM(input_size, hidden_size, n_layer, bidirectional=n_direction == 2)
x = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(n_layer*n_direction, batch_size, hidden_size)
c0 = torch.randn(n_layer*n_direction, batch_size, hidden_size)
o, (h, c) = lstm(x, (h0, c0))
print(o.shape)
print(h.shape)
print(c.shape)
assert o.shape == (seq_len, batch_size, n_direction*hidden_size)  # 注意 n_direction*hidden_size 有先后
assert h.shape == (n_direction*n_layer, batch_size, hidden_size)  # 注意 n_direction*n_layer 有先后
assert c.shape == (n_direction*n_layer, batch_size, hidden_size)  # 注意 n_direction*n_layer 有先后
