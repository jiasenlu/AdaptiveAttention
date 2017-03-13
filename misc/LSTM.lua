require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  table.insert(inputs, nn.Identity()()) -- indices giving the image feature  
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local img_fc = inputs[2]

  local x, input_size_L, i2h, fake_region, atten_region
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+2]
    local prev_c = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
      local w2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='t2h_'..L}
      local v2h = nn.Linear(input_size_L, 4 * rnn_size)(img_fc):annotate{name='v2h_'..L}
      i2h = nn.CAddTable()({w2h, v2h})
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
      i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    end

    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })

    local tanh_nex_c = nn.Tanh()(next_c)
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate,tanh_nex_c})
    if L == n then
      if L==1 then
        local w2h = nn.Linear(input_size_L, 1 * rnn_size)(x)
        local v2h = nn.Linear(input_size_L, 1 * rnn_size)(img_fc) 
        i2h = nn.CAddTable()({w2h, v2h})
      else
        i2h = nn.Linear(input_size_L, rnn_size)(x)
      end      
      local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
      local n5 = nn.CAddTable()({i2h, h2h})
      
      fake_region = nn.CMulTable()({nn.Sigmoid()(n5), tanh_nex_c})
    end

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = nn.Identity()(outputs[#outputs])
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  if dropout > 0 then fake_region = nn.Dropout(dropout)(fake_region) end

  table.insert(outputs, top_h)
  table.insert(outputs, fake_region)
  return nn.gModule(inputs, outputs)
end

return LSTM

