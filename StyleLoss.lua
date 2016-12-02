require 'nn'
require 'torch'
require 'utility'
dbg = require 'debugger'


local StyleLoss, parent = torch.class('nn.StyleLoss','nn.Module')


function StyleLoss:__init(strength, target)
	parent:__init(self)
	self.strength = strength
	self.target = target
	self.crit = nn.MSECriterion()
	self.Gram = GramModule()
	self.pred = nil 		-- the calulated GramMtraix from input
	self.loss = 0
	self.mode ='active'
end



function StyleLoss:updateOutput(input)
	if self.mode =='active' then
		self.pred = self.Gram:forward(input)
		self.pred:div(input:nElement())
		self.loss = self.crit:forward(self.pred, self.target) * self.strength
	else
		print('-- skipping StyleLoss layer --')
	end
	self.output = input
	return self.output
end



function StyleLoss:updateGradInput(input, gradOutput)
	local gradCrit = nil
	if self.mode == 'active' then
		gradCrit = self.crit:backward(self.pred, self.target)
		gradCrit:div(input:nElement())
		self.gradInput = self.Gram:backward(input, gradCrit) * self.strength
	end
	self.gradInput:add(gradOutput)
	return self.gradInput
end

function StyleLoss:setMode(mode)
	if mode~='active' and mode~='none' then
		error("StyleLoss can only be 'active' or 'none'! ")
	end
	self.mode = mode
end

