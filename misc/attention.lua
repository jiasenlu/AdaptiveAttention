require 'nn'
require 'nngraph'

local attention = {}
function attention.attention(input_size, rnn_size, output_size, dropout)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- top_h
  table.insert(inputs, nn.Identity()()) -- fake_region
  table.insert(inputs, nn.Identity()()) -- conv_feat
  table.insert(inputs, nn.Identity()()) -- conv_feat_embed

  local h_out = inputs[1]
  local fake_region = inputs[2]
  local conv_feat = inputs[3]
  local conv_feat_embed = inputs[4]

  local fake_region = nn.ReLU()(nn.Linear(rnn_size, input_size)(fake_region))
  -- view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
  if dropout > 0 then fake_region = nn.Dropout(dropout)(fake_region) end
  
  local fake_region_embed = nn.Linear(input_size, input_size)(fake_region)

  local h_out_linear = nn.Tanh()(nn.Linear(rnn_size, input_size)(h_out))
  if dropout > 0 then h_out_linear = nn.Dropout(dropout)(h_out_linear) end

  local h_out_embed = nn.Linear(input_size, input_size)(h_out_linear)

  local txt_replicate = nn.Replicate(50,2)(h_out_embed)

  local img_all = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region), conv_feat})
  local img_all_embed = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region_embed), conv_feat_embed})

  local hA = nn.Tanh()(nn.CAddTable()({img_all_embed, txt_replicate}))
  if dropout > 0 then hA = nn.Dropout(dropout)(hA) end
  local hAflat = nn.Linear(input_size,1)(nn.View(input_size):setNumInputDims(2)(hA))  
  local PI = nn.SoftMax()(nn.View(-1, 50):setNumInputDims(2)(hAflat))

  local probs3dim = nn.View(1,-1):setNumInputDims(1)(PI)
  local visAtt = nn.MM(false, false)({probs3dim, img_all})
  local visAttdim = nn.View(input_size):setNumInputDims(2)(visAtt)
  local atten_out = nn.CAddTable()({visAttdim, h_out_linear})

  local h = nn.Tanh()(nn.Linear(input_size, input_size)(atten_out))
  if dropout > 0 then h = nn.Dropout(dropout)(h) end
  local proj = nn.Linear(input_size, output_size)(h)

  local logsoft = nn.LogSoftMax()(proj)
  --local logsoft = nn.SoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
return attention
