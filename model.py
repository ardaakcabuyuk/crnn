import torch.nn as nn

class CRNN(nn.Module):

    def __init__(self, channels, height, width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, use_leaky_relu=False):

        super(CRNN, self).__init__()

        self.cnn, (output_channels, output_height, output_width) = \
            self.cnn_backbone(channels, height, width, use_leaky_relu)

        self.map_to_sequential = nn.Linear(output_channels *  output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def cnn_backbone(self, channels, height, width, use_leaky_relu):
        channels = [channels, 64, 128, 256, 256, 512, 512, 512]
        kernels = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def convolution_relu(i, batch_norm=False):
            # input shape: (batch size, input_channels, height, width)
            input_channels = channels[i]
            output_channels = channels[i + 1]

            cnn.add_module('conv-{}'.format(i), nn.Conv2d(input_channels, output_channels, kernels[i], strides[i], paddings[i]))

            if batch_norm:
                cnn.add_module('batchnorm-{}'.format(i), nn.BatchNorm2d(output_channels))

            if use_leaky_relu:
                relu = nn.LeakyReLU(0.2, inplace = True)
            else:
                relu = nn.ReLU(inplace = True)

            cnn.add_module('relu-{}'.format(i), relu)


        # size of image: (channels, height, width)

        convolution_relu(0)
        cnn.add_module('maxpool-0', nn.MaxPool2d(kernel_size = 2, stride = 2))
        # (64, height // 2, width // 2)

        convolution_relu(1)
        cnn.add_module('maxpool-1', nn.MaxPool2d(kernel_size = 2, stride = 2))
        # (128, height // 4, width // 4)

        convolution_relu(2)
        convolution_relu(3)
        cnn.add_module('maxpool-2', nn.MaxPool2d(kernel_size = (2,1)))
        # (256, height // 8, width // 4)

        convolution_relu(4, batch_norm=True)
        convolution_relu(5, batch_norm=True)
        cnn.add_module('maxpool-3', nn.MaxPool2d(kernel_size = (2,1)))
        # (512, height // 16, width // 4)

        convolution_relu(6)
        # (512, height // 16 - 1, width // 4 - 1)

        output_channels, output_height, output_width = channels[-1], height // 16 - 1, width // 4 - 1
        return cnn, (output_channels, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch_size, channels, height, width)

        convolution = self.cnn(images)
        batch_size, channels, height, width = convolution.size()

        convolution = convolution.view(batch_size, channels * height, width)
        convolution = convolution.permute(2, 0, 1) # (width, batch_size, features)

        sequential = self.map_to_sequential(convolution)

        recurrent, _ = self.rnn1(sequential)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output # shape: (sequential_length, batch_size, num_class)
