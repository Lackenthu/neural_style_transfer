# Neural Style Transfer - slow version
Computer Vision- Machine Learning

Make sure to download VGG-19 before running the script.

Basic Usage:

th neuralTransfer.lua -cuda \
  -content_weight 5e0 \
  -style_weight 1e2 \
  -content_image PATH/TO/CONTENT \
  -style_image PATH/TO/STYLE \
