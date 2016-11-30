require 'torch'
require 'image'
require 'nn'
require 'optim'
require 'loadcaffe'
require 'ContentLoss'
require 'StyleLoss'
require 'utility'

local dbg = require 'debugger'
require 'pretty-nn'

local cmd = torch.CmdLine()

cmd:option('-vgg_path',			'../models/',								'Path to model')
cmd:option('-content_image',	'../input/picture/tubingen.jpg',			'Path to input image')
cmd:option('-style_image',		'../input/art/starry_night.jpg',			'Path to input art')
cmd:option('-output_path',		'../output/',								'Path to output ')
cmd:option('-image_size',		512,										'Image crop size')
cmd:option('-cuda',				false,										'Using Cuda')
cmd:option('-content_layers',	'relu4_2',									'Layer for content')
cmd:option('-style_layers',		'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',	'Layer for style')
cmd:option('-content_weight',	5e0,										'Content weight')
cmd:option('-style_weight',		1e2,										'Style weight')

cmd:option('-lr',				0.1,										'learning rate')
cmd:option('-momentum',			0.5,										'momentum')
cmd:option('-max_iteration',	1000,										'Max Iteration number')
local opt = cmd:parse(arg)

local function main(opt)
	print("-- Entering body --")
	-- Variables
	torch.setdefaulttensortype('torch.FloatTensor')

	local vgg = loadcaffe.load(opt.vgg_path .. 'VGG_ILSVRC_19_layers_deploy.prototxt', opt.vgg_path .. 'VGG_ILSVRC_19_layers.caffemodel')
	local pic = image.load(opt.content_image,3)	-- content image
	local art = image.load(opt.style_image,3)	-- style image
	-- preprocess input 
	print('-- Preprocessing input image --')
	pic = preprocess(image.scale(pic,opt.image_size)):float()
	art = preprocess(image.scale(art,opt.image_size)):float()
	if opt.cuda == true then
		pic = pic:cuda()
		art = art:cuda()
	end
	local content_loss = {}
	local style_loss = {}
	local content_layers = opt.content_layers:split(',')
	local style_layers = opt.style_layers:split(',')
	local content_idx, style_idx = 10, 1

	-- preparing the revised VGG-19 net
	-- 1. replace all max-polling module with average-polling module
	-- 2. insert ContentLoss and StyleLoss module
	-- 3. drop all fully-connected module
	local net = nn.Sequential()
	print('-- Preparing model for training --')
	for i = 1, #vgg do 
		if (content_idx <= #content_layers or style_idx<= #style_layers) then
			local layer = vgg:get(i)

			-- replacing max-polling with average-polling
			if (torch.type(layer)=='nn.SpatialMaxPooling') then
				local kW ,kH, dW, dH = layer.kW, layer.kH, layer.dW, layer.dH
				local newLayer = nn.SpatialAveragePooling(kW,kH,dW,dH):float()
				if (opt.cuda==true) then
					newLayer:cuda()
				end
				net:add(newLayer)
			else
				if (opt.cuda==true) then
					net:add(layer:cuda())
				else
					net:add(layer)
				end
			end


			local name = layer.name
			--print(name)

			-- setting up ContentLoss layer
			if name == content_layers[content_idx] then
				print('-- Insert ContentLoss module at layer: ',name,' --')
				local target = net:forward(pic):clone()

				local content_mod = nn.ContentLoss(opt.content_weight,target)
				if opt.cuda == true then
					content_mod:cuda()
				end
				net:add(content_mod)
				table.insert(content_loss,content_mod)
				content_idx = content_idx+1
			end


			
			-- setting up StyleLoss layer
			if name == style_layers[style_idx] then
				print('-- Insert StyleLoss module at layer: ',name,' --')
				local feature = net:forward(art):clone()
				local gram = GramModule()
				if opt.cuda == true then
					gram:cuda()
				end
				local targetG = gram:forward(feature)
				targetG:div(feature:nElement())
				dbg()
				local style_mod = nn.StyleLoss(opt.style_weight,targetG)
				if opt.cuda == true then
					style_mod:cuda()
				end
				net:add(style_mod)
				table.insert(style_loss,style_mod)
				style_idx = style_idx+1
			end
		end
	end
	print('-- Model finished --')
	vgg = nil
	collectgarbage()
	-- prepare white noise image
	local img = torch.randn(pic:size()):float():mul(0.001)
	if opt.cuda==true then
		img = img:cuda()
	end

	local y = net:forward(img)
	local dy = torch.Tensor(#y):zero()
	--local dy = img.new(#y):zero()  
	-- ??? is img.new(#y):zero() == torch.Tensor(#y):zero() ???
	--local dy = torch.Tensor(#y):zero()





	--  training config
	local num_iter = 0


	local function feval(x)
		printState(num_iter)

		num_iter = num_iter+1
		net:forward(x)
		local loss = 0
		local closs = 0
		local sloss = 0
		local grad = net:updateGradInput(x, dy)

		for _, mod in ipairs(content_loss) do
			closs = closs+ mod.loss
		end
		for _, mod in ipairs(style_loss) do
			-- 5 active StyleLoss layer, each weights 0.2
			sloss = sloss+ mod.loss/5
		end
		loss = closs + sloss
		print(loss)
		return loss, grad:view(-1)
	end

	-- optim configuration
	local confit = {
		maxIter = opt.max_iteration
	}

	print('-- training using LBFGS optim --')
	local x, losses = optim.lbfgs(feval,img,config)
	print('-- training finished! --')
	x = depreprocess(x)
	image.save(opt.output_path..'output.jpg',x)
	print('The End!')

end




function printState(iter)
	if (iter % 10) ==0 then
		print('-- Iteration: ',iter,'/',opt.max_iteration,' --')
	end
end





print("Enter main")
main(opt)




