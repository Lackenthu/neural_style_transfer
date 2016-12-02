require 'nn'
require 'torch'
require 'image'
-- return a module that calculate a C*C matrix from a C*H*W matrix
function GramModule()
	local mod = nn.Sequential()
	mod:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
	concat:add(nn.Identity())
	concat:add(nn.Identity())
	mod:add(concat)
	mod:add(nn.MM(false, true))
	return mod
end


-- preprocess the image for VGG-19
-- 1. convert the image from RGB to BGR,
-- 2. convert the scale from [0,1] to [0,255]
-- 3. subtract the mean pixel
function preprocess(img)
	local pic_mean = torch.FloatTensor{103.939, 116.779, 123.68}
	local bgr = torch.LongTensor{3,2,1}
	img = img:index(1,bgr):mul(255)
	pic_mean = pic_mean:view(3,1,1):expandAs(img)
	img:add(-1,pic_mean)
	return img
end


-- reverse the preprocessing procedure
function depreprocess(img)
	local pic_mean = torch.FloatTensor{103.939, 116.779, 123.68}
	local rgb = torch.LongTensor{3,2,1}
	pic_mean = pic_mean:view(3,1,1):expandAs(img)
	img:add(pic_mean)
	img = img:index(1,rgb):div(255)
	return img
end
