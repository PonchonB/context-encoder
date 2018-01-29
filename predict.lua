require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

test_dir = 'images/'
filename_pred = 'out.png'
gpu = 0
inputSize = 128
maskValues = {2*117.0/255.0 - 1.0, 2*104.0/255.0 - 1.0, 2*123.0/255.0 - 1.0}
-- maskValues = {2*117.0/255.0 - 1.0, 2*104.0/255.0 - 1.0, 2*123.0/255.0 - 1.0}
opt = {
    input_path = 'input_tmp.png', -- path to input image
    batchSize = 1,        -- number of samples to produce
    net = 'models/inpaintCenter/imagenet_inpaintCenter.t7', -- path to the generator network
--    imDir = 'results/',            -- directory containing pred_center
    output_path = 'output_tmp.png',     -- name of the file saved
    gpu = 0,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    manualSeed = 222,        -- 0 means random seed
    overlapPred = 4,       -- overlapping edges of center with context
    
}


for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

opt.output_path = opt.input_path

input_path = 'images/' .. opt.input_path
output_path = 'results/' .. opt.output_path

--load pre-trained network trained on paris_street_view images
net = torch.load(opt.net)

net:apply(function(m) if m.weight then
    m.gradWeight = m.weight:clone():zero(); -- put grad weight to zero
    m.gradBias = m.bias:clone():zero(); end end)
net:evaluate()

-- init input_ctx tensor out output_ctx tensor


-- image_ctx = torch.Tensor(opt.nc, inputSize, inputSize)
input_ctx = torch.Tensor(opt.nc, inputSize, inputSize)
output_ctx = torch.Tensor(opt.nc, inputSize, inputSize)

-- load input image
imPath = input_path
input = image.load(input_path, opt.nc, 'float') -- pixel value in 0..1
input = image.scale(input, inputSize, inputSize)
input:mul(2):add(-1) -- pixel value in -1..1

-- itorch.image(input)

-- copy input image to input_ctx tensor and output_ctx tensor
input_ctx:copy(input)
output_ctx:copy(input)
print('Loaded Image Block: ', input_ctx:size(1)..' x '..input_ctx:size(2) ..' x '..input_ctx:size(3))

-- keep real_center in a separate tensor real_center
real_center = input[{{},{1 + inputSize/4, inputSize/2 + inputSize/4},{1 + inputSize/4, inputSize/2 + inputSize/4}}]:clone()

-- itorch.image(real_center)

-- fill center region with mean value
input_ctx[{{1},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = maskValues[1]
input_ctx[{{2},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = maskValues[2]
input_ctx[{{3},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = maskValues[3]

-- itorch.image(input_ctx)

-- run Context-Encoder to inpaint center
pred_center = net:forward(input_ctx)
print('Prediction: size: ', pred_center:size(1)..' x '..pred_center:size(2) ..' x '..pred_center:size(3)..' x '..pred_center:size(4))
print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

-- paste predicted center in output_ctx tensor
output_ctx[{{},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}]:copy(pred_center[{{},{},{1 + opt.overlapPred, inputSize/2 - opt.overlapPred},{1 + opt.overlapPred, inputSize/2 - opt.overlapPred}}])

-- re-transform scale back to normal
input_ctx:add(1):mul(0.5)
output_ctx:add(1):mul(0.5)

-- itorch.image(output_ctx)
-- image_ctx:add(1):mul(0.5)
-- pred_center:add(1):mul(0.5)
-- real_center:add(1):mul(0.5)

image.save(output_path, output_ctx)
