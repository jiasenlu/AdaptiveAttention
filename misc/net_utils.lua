local utils = require 'misc.utils'
local net_utils = {}



-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn_conv_fix(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 10)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  return cnn_part
end



-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn_conv(cnn, opt)
  local layer_num_start = utils.getopt(opt, 'layer_num_start', 11)
  local layer_num = utils.getopt(opt, 'layer_num', 37)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = layer_num_start, layer_num do
    local layer = cnn:get(i)
    cnn_part:add(layer)
  end

  return cnn_part
end

function net_utils.build_residual_cnn_conv_fix(cnn, opt)
  local layer_num = utils.getopt(opt, 'start_layer_num', 6)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num-1 do
    local layer = cnn:get(i)
    cnn_part:add(layer)
  end
  --cnn_part:add(nn.View(512, -1):setNumInputDims(3))
  --cnn_part:add(nn.Transpose({2,3}))
  return cnn_part
end


-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_residual_cnn_conv(cnn, opt)
  local start_layer_num = utils.getopt(opt, 'start_layer_num', 6)
  local layer_num = utils.getopt(opt, 'layer_num', 8)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = start_layer_num, layer_num do
    local layer = cnn:get(i)
    cnn_part:add(layer)
  end

  --cnn_part:add(nn.View(512, -1):setNumInputDims(3))
  --cnn_part:add(nn.Transpose({2,3}))
  return cnn_part
end


function net_utils.transform_cnn_conv(nDim)

  local cnn_part = nn.Sequential()

  cnn_part:add(nn.View(nDim, -1):setNumInputDims(3))
  cnn_part:add(nn.Transpose({2,3}))
  return cnn_part
end


-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn_fc(cnn, opt)
  local layer_num_start = utils.getopt(opt, 'layer_num', 38)
  local layer_num_end = utils.getopt(opt, 'layer_num', 43)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = layer_num_start, layer_num_end do
    local layer = cnn:get(i)

    cnn_part:add(layer)
  end

  return cnn_part
end

function net_utils.build_residual_cnn_fc(cnn, opt)
  local layer_num_start = utils.getopt(opt, 'layer_num', 9)
  local layer_num_end = utils.getopt(opt, 'layer_num', 10)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = layer_num_start, layer_num_end do
    local layer = cnn:get(i)

    cnn_part:add(layer)
  end

  return cnn_part
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

local layer, parent = torch.class('nn.FeatExpanderConv', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  local s = input:size(2)
  local d = input:size(3)
  self.output:resize(input:size(1)*self.n, s, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, s, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end


function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local count = {}
  for i=1,N do
    local tmp = 0
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      tmp = tmp + 1
      txt = txt .. word
    end
    --txt = txt .. '.'
    table.insert(count, tmp)

    table.insert(out, txt)
  end
  return out, count
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

function net_utils.clone_list_all(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    local new_sub = {}
    for m,n in pairs(v) do
      new_sub[m] = n:clone()
    end
    new[k] = new_sub
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, opt)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local id = utils.getopt(opt, 'id', 1)
  local dataset = utils.getopt(opt, 'dataset','coco')

  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  print('./misc/call_python_caption_eval.sh val' .. id .. '.json annotations/' ..dataset..'.json')
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json annotations/' ..dataset..'.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

function net_utils.init_noise(graph, batch_size)
  if batch_size == nil then
    error('please provide valid batch_size value')
  end
  for i, node in pairs(graph:listModules()) do
    local layer = graph:get(i)
    local t = torch.type(layer)
    if t == 'nn.DropoutFix' then
      layer:init_noise(batch_size)
    end
  end
end

function net_utils.deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = net_utils.deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function net_utils.setBNGradient0(graph)
   -- setting the gradient of BN to be zero
  local BNlayers = graph:findModules('nn.SpatialBatchNormalization')
  for i, node in pairs(BNlayers) do
    node.gradWeight:zero()
    node.gradBias:zero()
  end
end


return net_utils