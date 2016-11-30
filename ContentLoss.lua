
require 'nn'
require 'torch'


local ContentLoss, parent = torch.class('nn.ContentLoss','nn.Module')


function ContentLoss:__init(strength,target)
	parent.__init(self)
	self.strength = strength
	self.target = target
	self.loss = 0
	self.crit = nn.MSECriterion()
	self.mode = 'active'
end


function ContentLoss:updateOutput(input)
	if self.mode=='active' and input:nElement()==self.target:nElement() then
		self.loss = self.crit:forward(input,self.target) * self.strength
	else
		print('-- skipping ContentLoss layer --')
	end
	self.output = input
	return self.output
end


function ContentLoss:updateGradInput(input, gradOutput)
	if self.mode=='active' and input:nElement()==self.target:nElement() then
		self.gradInput = self.crit:backward(input,self.target) *self.strength
	end
	self.gradInput:add(gradOutput)
	return self.gradInput
end

function ContentLoss:setMode(mode)
	if mode~='active' and mode~='none' then
		error("ContentLoss can only be 'active' or 'none'! ")
	end
	self.mode = mode
end