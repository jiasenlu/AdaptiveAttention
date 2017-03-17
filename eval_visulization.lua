require 'torch'
require 'nn'
require 'nngraph'
-- local imports
require 'visu.DataLoaderResNetEval'
local utils = require 'misc.utils'
require 'visu.LanguageModel_visu'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'gnuplot'
require 'xlua'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')


-- Model settings
--[[
cmd:option('-dataset','flickr30k','')
cmd:option('-input_h5','/data/flickr30k/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/data/flickr30k/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_model','../image_model/resnet-152.t7','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
]]--

cmd:option('-input_h5','/data/coco/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/data/coco/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_model','../image_model/resnet-152.t7','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-checkpoint_path', 'save/coco_val_1', 'folder to save checkpoints into (empty = this folder)')

--[[
cmd:option('-input_h5','/data/coco/cocotalk_test.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/data/coco/cocotalk_test.json','path to the json file containing additional info and vocab')
cmd:option('-input_vocab_json','/data/coco/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_model','../image_model/resnet-152.t7','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
]]--
cmd:option('-start_from', 'model_id1_36.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-beam_size', 3, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
--cmd:option('-checkpoint_path', 'save/flickr30k_512x1_1', 'folder to save checkpoints into (empty = this folder)')

cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers',1,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-batch_size',10,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')

cmd:option('-fc_size',2048,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-conv_size',2048,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 3, 'how often to save a model checkpoint?')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
--torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  --cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, neighbor_h5 = opt.nn_neighbor, 
                  batch_size = opt.batch_size, seq_per_img = opt.seq_per_img, thread_num = opt.thread_num}
--local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, vocab_json_file = opt.input_vocab_json,neighbor_h5 = opt.nn_neighbor, 
--                  batch_size = opt.batch_size, seq_per_img = opt.seq_per_img, thread_num = opt.thread_num}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
-- create protos from scratch
-- intialize language model
local lmOpt = {}
lmOpt.vocab_size = loader:getVocabSize()
lmOpt.input_encoding_size = opt.input_encoding_size
lmOpt.rnn_size = opt.rnn_size
lmOpt.num_layers = opt.num_layers
lmOpt.dropout = opt.drop_prob_lm
lmOpt.seq_length = loader:getSeqLength()
lmOpt.batch_size = opt.batch_size * opt.seq_per_img
lmOpt.fc_size = opt.fc_size
lmOpt.conv_size = opt.conv_size

local loaded_checkpoint
if opt.start_from ~= '' then -- just copy to gpu1 params
  local loaded_checkpoint_path = path.join(opt.checkpoint_path, opt.start_from)
  print(loaded_checkpoint_path)
  loaded_checkpoint = torch.load(loaded_checkpoint_path)
end

-- iterate over different gpu
local protos = {}

protos.lm = nn.LanguageModel(lmOpt):cuda()
-- initialize the ConvNet 
if opt.start_from ~= '' then -- just copy to gpu1 params
  protos.cnn_conv_fix = loaded_checkpoint.protos.cnn_conv_fix:cuda()
  protos.cnn_conv = loaded_checkpoint.protos.cnn_conv:cuda()
  protos.cnn_fc = loaded_checkpoint.protos.cnn_fc:cuda()
else
  local cnn_raw = torch.load(opt.cnn_model)

  protos.cnn_conv_fix = net_utils.build_residual_cnn_conv_fix(cnn_raw, 
                      {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()

  protos.cnn_conv = net_utils.build_residual_cnn_conv(cnn_raw, 
                      {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()

  protos.cnn_fc = net_utils.build_residual_cnn_fc(cnn_raw, 
                      {backend = cnn_backend}):cuda()
end
protos.expanderConv = nn.FeatExpanderConv(opt.seq_per_img):cuda()
protos.expanderFC = nn.FeatExpander(opt.seq_per_img):cuda()
protos.transform_cnn_conv = net_utils.transform_cnn_conv(opt.conv_size):cuda()
-- criterion for the language model
protos.crit = nn.LanguageModelCriterion():cuda()

params, grad_params = protos.lm:getParameters()
cnn1_params, cnn1_grad_params = protos.cnn_conv:getParameters()

print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN_conv: ', cnn1_params:nElement())

assert(params:nElement() == grad_params:nElement())
assert(cnn1_params:nElement() == cnn1_grad_params:nElement())

if opt.start_from ~= '' then -- just copy to gpu1 params
  params:copy(loaded_checkpoint.lmparam)
end

protos.lm:createClones()

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function evaluate_split(split, evalopt)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', -1)

  print('=> evaluating ...')

  -- setting to the evaluation mode, use only the first gpu
  protos.cnn_conv:evaluate()
  protos.cnn_fc:evaluate()
  protos.lm:evaluate()
  protos.cnn_conv_fix:evaluate()

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  local imgId_cell = {}

  local nbatch = math.ceil(val_images_use / opt.batch_size)
  if val_images_use == -1 then
    nbatch = loader:getnBatch(split)
  end

  loader:init_rand(split)
  loader:reset_iterator(split)

  local atten_out_all = torch.FloatTensor(loader:getSeqLength()+1, 5*nbatch*opt.batch_size, 50):zero()
  --for n, data in loader:run({split = split, size_image_use = val_images_use}) do
  for n = 1, nbatch do
    local data = loader:run({split = split, size_image_use = val_images_use})
    xlua.progress(n,nbatch)

    -- convert the data to cuda
    data.images = data.images:cuda()
    data.labels = data.labels:cuda()

    -- forward the model to get loss
    local feats_conv_fix = protos.cnn_conv_fix:forward(data.images)

    local feats_conv = protos.cnn_conv:forward(feats_conv_fix)
    local feat_conv_t = protos.transform_cnn_conv:forward(feats_conv)
    local feats_fc = protos.cnn_fc:forward(feats_conv)    

    local expanded_feats_conv = protos.expanderConv:forward(feat_conv_t)
    local expanded_feats_fc = protos.expanderFC:forward(feats_fc)
    
    local logprobs, atten = protos.lm:forward({expanded_feats_conv, expanded_feats_fc, data.labels})
    --local loss = protos.crit:forward({logprobs, data.labels})
    --loss_sum = loss_sum + loss
    --loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local sampleOpt = {beam_size = opt.beam_size}
    --local seq, atten = protos.lm:sample({feat_conv_t, feats_fc, vocab}, sampleOpt)
    local sents, count = net_utils.decode_sequence(vocab,  data.labels)

    local s = (n-1)*opt.batch_size*5+1
    atten_out_all:narrow(2,s,opt.batch_size*5):copy(atten)

    for k=1,#sents do
      local idx = math.floor((k-1)/5)+1
      local img_id = data.img_id[idx]
      local entry
      --if imgId_cell[img_id] == nil then -- make sure there are one caption for each image.
        --imgId_cell[img_id] = 1
        local prob_tmp = {}
        for m = 1, count[k] do
          table.insert(prob_tmp, 1-atten[m][k][1])
        end
        entry = {image_id = img_id, caption = sents[k], prob = prob_tmp}
        table.insert(predictions, entry)
      --end
    end
  end
  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, {id = opt.id, dataset = opt.dataset})
  end

  return predictions, lang_stats, atten_out_all
end

local split_predictions, lang_stats, atten_out_all = evaluate_split('test', {val_images_use = opt.val_images_use, verbose = opt.verbose})

if lang_stats then
  print(lang_stats)
end

utils.write_json('visu_gt_test.json', split_predictions)
torch.save('atten_gt_test_1.t7', atten_out_all)


