require 'image'
require 'hdf5'

cjson=require('cjson')

function read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end
local path = 'flickr30k_gt/'

local atten = torch.load(path .. 'atten.t7')
local caption = read_json(path ..'visu.json')

local json_file = read_json('/data/flickr30k/cocotalk.json')
local id2path = {}
for i = 1,5000 do
  id = json_file['images'][i]['id']
  path = json_file['images'][i]['file_path']
  id2path[id] = path
end


local num = #caption
local cap_all = {}
--local scale_map = torch.FloatTensor(num, 23, 224, 224)
local atten_weight = torch.FloatTensor(num, 23)
local atten_map_original = torch.FloatTensor(num, 23, 7, 7)

f = io.open('flickr30k.txt', 'w')
for t = 1, num do
    local cap = caption[t]['caption']
    local img_id = caption[t]['image_id']
    local atten_map = atten[{{},{t},{}}]:contiguous():view(23,50)

    for i = 1, atten_map:size(1) do
        local map = atten_map:sub(i,i, 2, 50):view(7,7)
        atten_map_original:sub(t,t,i,i):copy(map)
        atten_weight:sub(t,t,i,i):copy(atten_map:sub(i,i, 1, 1))
        --scale_map:sub(t, t, i,i):copy(image.scale(map, 224, 224, 'bicubic'))
    end

    f:write(tostring(id2path[img_id]))
    f:write('\t')
    f:write(cap)
    f:write('\n')
    table.insert(cap_all, {cap, img_id})
end
f.close()

print({atten_weight})
local myFile = hdf5.open('atten_img.h5', 'w')
--myFile:write('map', scale_map)
myFile:write('atten_weight', atten_weight)
myFile:write('atten_map_original', atten_map_original)
myFile:close()
write_json('cap.json', cap_all)



