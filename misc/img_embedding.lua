require 'nn'
require 'nngraph'

local img_embedding = {}

function img_embedding.img_embedding(hidden_size, fc_size, conv_size, conv_num, dropout)
	local inputs = {}
	local outputs = {}

	 table.insert(inputs, nn.Identity()()) -- image feature 
  	table.insert(inputs, nn.Identity()()) 

  	local conv_feat = inputs[1]
    local fc_feat = inputs[2]

    -- embed the fc7 feature -- dropout here? 
    local fc_feat_out = nn.ReLU()(nn.Linear(fc_size, hidden_size)(fc_feat))
    if dropout > 0 then fc_feat_out = nn.Dropout(dropout)(fc_feat_out) end

    -- embed the conv feature
    local conv_feat_embed = nn.Linear(conv_size, hidden_size)(nn.View(conv_size):setNumInputDims(2)(conv_feat))
    
    local conv_feat_out = nn.ReLU()(conv_feat_embed)
    if dropout > 0 then conv_feat_out = nn.Dropout(dropout)(conv_feat_out) end

    local conv_feat_back = nn.View(-1, conv_num, hidden_size)(conv_feat_out)


  	local img_feat_dim = nn.View(hidden_size):setNumInputDims(2)(conv_feat_back)
  	local embed_feat = nn.Linear(hidden_size, hidden_size)(img_feat_dim)
  	local embed_feat_out = nn.View(-1, conv_num, hidden_size)(embed_feat)

 
  	table.insert(outputs, conv_feat_back)
  	table.insert(outputs, embed_feat_out)
    table.insert(outputs, fc_feat_out)

  	return nn.gModule(inputs, outputs)

end

return img_embedding

