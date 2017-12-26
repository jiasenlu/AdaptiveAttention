require 'nn'
require 'misc.LookupTableMaskZero'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'

local attention = require 'misc.attention'
local img_embedding = require 'misc.img_embedding'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.n_rnn_layer = utils.getopt(opt, 'n_rnn_layer', 1)

  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)

  self.fc_size = utils.getopt(opt, 'fc_size', 4096)
  self.conv_size = utils.getopt(opt, 'conv_size', 512)

  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')

  print('rnn_size: ' ..  self.rnn_size .. ' num_layers: ' .. self.num_layers)
  print('input_encoding_size: ' ..  self.input_encoding_size)
  print('dropout rate: ' .. dropout)

  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = LSTM.lstm(self.input_encoding_size, self.rnn_size, self.num_layers, dropout)

  self.lookup_table = nn.Sequential()
                  :add(nn.LookupTableMaskZero(self.vocab_size+1, self.input_encoding_size))
                  :add(nn.ReLU())
                  :add(nn.Dropout(dropout))

  self.img_embedding = img_embedding.img_embedding(self.input_encoding_size, self.fc_size,  self.conv_size, 49, dropout)

  self.attention = attention.attention(self.input_encoding_size, self.rnn_size, self.vocab_size+1, dropout)

  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end


function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  self.attentions = {self.attention}
  for t=2,self.seq_length+1 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
    self.attentions[t] =  self.attention:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end


function layer:getModulesList()
  return {self.core, self.lookup_table, self.img_embedding, self.attention}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()
  local p3,g3 = self.img_embedding:parameters()
  local p4,g4 = self.attention:parameters()


  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end
  for k,v in pairs(p4) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end
  for k,v in pairs(g4) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.attentions) do v:training() end
  self.img_embedding:training()

end

function layer:evaluate()
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.attentions) do v:evaluate() end
  self.img_embedding:evaluate()
end

function layer:sample(inputs, opt)
  local conv = inputs[1]
  local fc = inputs[2] 
  local ix_to_word = inputs[3]


  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
    
  local temperature = utils.getopt(opt, 'temperature', 1.0)

  local batch_size = fc:size(1)

  if sample_max == 1 and beam_size > 1 then return self:sample_beam(inputs, opt) end -- indirection for beam search

  self:_createInitState(batch_size)
  local state = self.init_state
  
  local img_input = {conv, fc}
  local conv_feat, conv_feat_embed, fc_embed = table.unpack(self.img_embedding:forward(img_input))

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

  local logprobs  -- logprobs predicted in last time step
  local x_xt

  for t=1,self.seq_length+1 do
    local xt, it, sampleLogprobs
    if t == 1 then
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    local inputs = {xt,fc_embed, table.unpack(state)}
    local out = self.core:forward(inputs)
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end

    local h_out = out[self.num_state+1]
    local p_out = out[self.num_state+2]

    local atten_input = {h_out, p_out, conv_feat, conv_feat_embed}
    logprobs = self.attention:forward(atten_input)

  end
  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end


function layer:sample_beam(inputs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)

  local conv = inputs[1]
  local fc = inputs[2] 
  local ix_to_word = inputs[3]

  local batch_size = fc:size(1)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local img_input = {conv, fc}
  local conv_feat, conv_feat_embed, fc_embed = table.unpack(self.img_embedding:forward(img_input))

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local seqLogprobs_sum = torch.FloatTensor(batch_size)

  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    local imgk = fc_embed[{ {k,k} }]:expand(beam_size, self.input_encoding_size) -- k'th image feature expanded out
    local conv_feat_k = conv_feat[{ {k,k} }]:expand(beam_size, conv_feat:size(2), self.input_encoding_size) -- k'th image feature expanded out
    local conv_feat_embed_k = conv_feat_embed[{ {k,k} }]:expand(beam_size, conv_feat_embed:size(2), self.input_encoding_size) -- k'th image feature expanded out

    for t=1,self.seq_length+1 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 2 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end

        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        it = beam_seq[t-1]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt, imgk, table.unpack(state)}
      local out = self.core:forward(inputs)
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
      local h_out = out[self.num_state+1]
      local p_out = out[self.num_state+2]
      local atten_input = {h_out, p_out, conv_feat_k, conv_feat_embed_k}
      logprobs = self.attention:forward(atten_input)

    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
    seqLogprobs_sum[k]=done_beams[1].p
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs_sum
end

function layer:updateOutput(input)
  local conv = input[1]
  local fc = input[2]  
  local seq = input[3]

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)

  self:_createInitState(batch_size)

  -- first get the nearest neighbor representation.
  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1):zero()

  self.img_input = {conv, fc}
  self.conv_feat, self.conv_feat_embed, self.fc_embed = table.unpack(self.img_embedding:forward(self.img_input))

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.atten_inputs = {}
  --self.x_inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency

  for t = 1,self.seq_length+1 do
    local can_skip = false
    local xt
    if t == 1 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_table:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        can_skip = true 
      end

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

    if not can_skip then
      -- construct the inputs
      self.inputs[t] = {xt, self.fc_embed, table.unpack(self.state[t-1])}
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      -- insert the hidden state
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      local h_out = out[self.num_state+1]
      local p_out = out[self.num_state+2]

      --forward the attention
      self.atten_inputs[t] = {h_out, p_out, self.conv_feat, self.conv_feat_embed}
      local atten_out = self.attentions[t]:forward(self.atten_inputs[t])

      self.output:narrow(1,t,1):copy(atten_out)
      self.tmax = t
    end
  end

  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dconv, dconv_embed, dfc-- grad on input images

  local batch_size = self.output:size(2)
  -- go backwards and lets compute gradients
  local dstate = self.init_state -- this works when init_state is all zeros

  for t=self.tmax,1,-1 do

    local d_atten = self.attentions[t]:backward(self.atten_inputs[t], gradOutput[t])
    if not dconv then dconv = d_atten[3] else dconv:add(d_atten[3]) end
    if not dconv_embed then dconv_embed = d_atten[4] else dconv_embed:add(d_atten[4]) end

    local dout = {}
    for k=1, self.num_state do table.insert(dout, dstate[k]) end
    table.insert(dout, d_atten[1])
    table.insert(dout, d_atten[2])

    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    
    local dxt = dinputs[1] -- first element is the input vector
    if not dfc then dfc = dinputs[2] else dfc:add(dinputs[2]) end

    dstate = {} -- copy over rest to state grad
    for k=3,self.num_state+2 do table.insert(dstate, dinputs[k]) end
    
    -- continue backprop of xt
    local it = self.lookup_tables_inputs[t]
    self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
  end

  -- backprob to the visual features.
  local dimgs_cnn, dfc_cnn = table.unpack(self.img_embedding:backward(self.img_input, {dconv, dconv_embed, dfc}))

  self.gradInput = {dimgs_cnn, dfc_cnn}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

function crit:updateOutput(inputs)
  local input = inputs[1]
  local seq = inputs[2]
  --local seq_len = inputs[3]

  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  self.gradInput:resizeAs(input):zero()
  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)
      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}]
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end
    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  
  return self.output
end

function crit:updateGradInput(inputs)
  return self.gradInput
end
