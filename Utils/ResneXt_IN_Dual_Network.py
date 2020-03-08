import keras.backend as K
import six
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, Lambda, concatenate, add, \
    GlobalAveragePooling3D, Dense, GlobalMaxPooling3D, Flatten, Dropout, MaxPooling3D, Reshape, Conv2D, MaxPooling2D
from keras.engine.topology import Layer

from SpatialPyramidPooling import SpatialPyramidPooling


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    # TODO 分组卷积
    # grouped_channels=16  strides=1
    # grouped_channels=32  strides=2
    # grouped_channels=64  strides=2

    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    group_list = []

    # 标准卷积，不执行
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):  # 8组
        # 根据channel维度进行分组
        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)
        # 执行lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]  channel_last  -1
        # input[:,:,:,:,c*64:(c+1)*64]  将128维切片成8组，每组16个滤波器，0:16  17:32  ...  113:128
        # lambda_1--8 9--16 17--24
        # 输入（9，9，37，128）  输出（9，9，37，16）这里的16是128的8组
        # lambda_25--32
        # 输入（9，9，37，256）  输出（9，9，37，32）这里的32是256的8组
        # lambda_25--32
        # 输入（5，5，19，512）  输出（5，5，19，64）这里的64是512的8组

        # 分组后各自卷积
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        # 3x3x3，16，s=（1，1, 1），SAME  conv2d_4--11 14--21 24--31
        # 输入（9，9，37，16）  输出（9，9，37，16）
        # 3x3x3，32，s=（2，2，2），SAME  conv2d_35--42
        # 输入（9，9，37，32）  输出（5，5，19，32）
        # 3x3x3，64，s=（2，2，2），SAME  conv2d_35--42
        # 输入（5，5，19，64）  输出（3，3，10，64）
        group_list.append(x)  # 将x存放在列表里，共8个

    # concat拼接
    group_merge = concatenate(group_list, axis=channel_axis)  # 将8个以channel维（最后一维）拼接
    # concatenate_1  （9，9，37，128）
    # concatenate_2  （5，5，19，256）
    # concatenate_3  （3，3，10，512）
    x = BatchNormalization(axis=channel_axis)(group_merge)

    return x


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    # TODO resnext模块

    init = input
    # （9，9，37，24） 128 s=1
    # （9，9，37，128） 256 s=2
    # （5，5，19，256） 512 s=2

    # 分组卷积的个数
    grouped_channels = int(filters / cardinality)  # 128/8=16  256/8=32  512/8=64
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    # TODO 右边shortcut
    # 判断底层
    if K.image_data_format() == 'channels_first':  # 不执行
        if init._keras_shape[1] != 2 * filters:
            init = Conv3D(filters * 2, (1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:  # 底层tf，执行下面的代码
        if init._keras_shape[-1] != 2 * filters:  # 使用1x1x1卷积改变尺寸，使得shortcut的channel维度和分组卷积的滤波器相同才可以add
            init = Conv3D(filters, (1, 1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            # padding为same时，只表示图像的数据不丢失，若步长为1，则输出图像的大小不变；若步长为2，则输出图像的大小为输入图像大小的一半。
            # 1x1x1，128，s=（1，1，1），SAME  conv2d_2
            # 输入（9，9，37，24）  输出（9，9，37，128）
            # 1x1x1，256，s=（2，2，2），SAME  conv2d_33
            # 输入（9，9，37，128）  输出（5，5，19，256）
            # 1x1x1，512，s=（2，2，2），SAME
            # 输入（5，5，19，256）  输出（3，3，10，512）
            init = BatchNormalization(axis=channel_axis)(init)  # 这里不加relu是因为在后面add之后一起激活

    # TODO 左边分组卷积之前的卷积，使用1x1改变卷积核个数
    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    # 1x1x1，128，s=1，SAME  conv2d_3
    # 输入（9，9，37，24）  输出（9，9，37，128）
    # 1x1x1，256，s=1，SAME  conv2d_34
    # 输入（9，9，37，128）  输出（9，9，37，256）
    # 1x1x1，512，s=1，SAME  conv2d_34
    # 输入（5，5，19，256）  输出（5，5，19，512）
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # TODO 分组卷积
    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    # grouped_channels=16  strides=1
    # 8组  输入（9，9，37，128）  输出（9，9，37，128）
    # grouped_channels=32  strides=2
    # 8组  输入（9，9，37，256）  输出（5，5，19，256）
    # grouped_channels=64  strides=2
    # 8组  输入（5，5，19，512）  输出（3，3，10，512）

    # TODO 左右连接
    x = add([init, x])  # add_1  残差连接shortcut
    # 输出（9，9，37，128）
    # 输出（5，5，19，256）
    # 输出（3，3，10，512）

    x = Activation('relu')(x)

    return x


def __initial_conv_block(input, weight_decay=5e-4):
    # TODO 初始化卷积层
    # x = __initial_conv_block(img_input, weight_decay)

    # 底层tf，channel最后
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    # 3x3x20，24，s=（1，1，5）
    x = Conv3D(32, (3, 3, 7), strides=(1, 1, 2), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), name="init_conv")(input)
    # 输入（11，11，200，1）  输出（9，9，37，24）
    x = BatchNormalization(axis=channel_axis, name="init_BN")(x)
    # x = Activation('relu', name="init_ReLU")(x)

    return x


def _bn_relu_spc(input):  # BN + ReLU（spectral）
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu_spc(**conv_params):  # CONV + BN + ReLU（spectral） 先CONV再BN和激活
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))  # 下采样步长
    init = conv_params.setdefault("init", "he_normal")  # He正态分布初始化方法，初始化权重函数名称
    border_mode = conv_params.setdefault("border_mode", "same")  # 补零
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))  # W正则化

    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu_spc(conv)

    return f


def _bn_relu_conv_spc(**conv_params):  # BN + ReLU + CONV  先BN和激活再CONV（改进方法）
    # residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))  # 子采样步长
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      padding=border_mode)(activation)

    return f


def _shortcut_spc(input, residual):  # shortcut 残差块
    # _shortcut_spc(input, residual)
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3] + 1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]  # 通道匹配

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual._keras_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1, 1),
                          strides=(stride_dim1, stride_dim2, stride_dim3),
                          kernel_initializer="he_normal", padding="valid",
                          kernel_regularizer=l2(0.0001))(input)  # 使用1*1CONV，使得shortcut和residual通道匹配
    return add([shortcut, residual])  # 输出 shortcut + residual


def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):  # repetitions = 1 --> i = 0
            init_subsample = (1, 1, 1)  # 残差块中的CONV的S
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)  # 残差块前的CONV的S
            # init_subsample = (1, 1, 1)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):
        # is_first_block_of_first_layer=True
        if is_first_block_of_first_layer:  # 如果是第一层的第一个块
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=l2(0.0001),
                           filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)  # CONV2
        else:  # 不是残差块外的CONV，直接运行这一个条件
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                      subsample=init_subsample)(input)
        # spectral的CONVBN 1*1*7,24
        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)  # CONV3
        return _shortcut_spc(input, residual)  # input + residual 第一个残差块的和

    return f


def _get_block(identifier):
    # block_fn_spc = _get_block(block_fn_spc)
    # block_fn = _get_block(block_fn)
    if isinstance(identifier, six.string_types):  # isinstance:判断是否是指定的格式
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def CNN_block(input, repetitions1=None, block_fn_spc=None):
    block_fn_spc = _get_block(block_fn_spc)
    conv1_spc = _conv_bn_relu_spc(nb_filter=24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=7,
                                  subsample=(1, 1, 2))(input)
    # 残差块
    block_spc = conv1_spc
    nb_filter = 32
    for i, r in enumerate(repetitions1):  # i=0, r=1
        # i是索引，从0开始
        # r是元素
        block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r,
                                        is_first_layer=(i == 0))(block_spc)  # is_first_layer=True, repetitions=1
        nb_filter *= 2

    return block_spc


def __create_res_next(nb_classes, img_input, cardinality=8, weight_decay=5e-4):
    # TODO 网络搭建
    # x_dense = __create_res_next(classes, input, cardinality, weight_decay)

    # 三层模块的滤波器个数
    # filters_list = [64, 128, 256, 512]  # 64, 128, 256, 512
    if cardinality == 6:
        filters_list = [48, 96, 192, 384]
    elif cardinality == 8:
        filters_list = [64, 128, 256, 512]
    elif cardinality == 10:
        filters_list = [80, 160, 320, 640]

    # TODO 初始化卷积层
    x = __initial_conv_block(img_input, weight_decay)  # img_input=（11，11，200，1）
    # 输入（11，11，200，1）  输出（9，9，37，24）

    # TODO CNN（SPC)
    x_spc = CNN_block(img_input, repetitions1=[1], block_fn_spc=basic_block_spc)

    # TODO 融合
    x_add = add([x, x_spc])

    x_add = Activation('relu')(x_add)  # (9,9,97,32)

    # TODO 第一个模块
    x_1 = __bottleneck_block(x_add, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)
    # filters_list[0]=128  cardinality=8  strides=1
    # 输入（9，9，97，32）  输出（9，9，97，64）

    # TODO 第二个模块
    x_2 = __bottleneck_block(x_1, filters_list[1], cardinality, strides=2, weight_decay=weight_decay)
    # filters_list[1]=256  cardinality=8  strides=2
    # 输入（9，9，97，64）  输出（5，5，49，128）

    # TODO 第三个模块
    x_3 = __bottleneck_block(x_2, filters_list[2], cardinality, strides=2, weight_decay=weight_decay)
    # filters_list[1]=512  cardinality=8  strides=2
    # 输入（5，5，49，128）  输出（3，3，25，256）

    # TODO 第四个模块
    x_4 = __bottleneck_block(x_3, filters_list[3], cardinality, strides=2, weight_decay=weight_decay)
    # 输入（3，3，25，256）  输出（2，2，13，512）

    drop = Dropout(0.5)(x_4)

    # pooling_regions = [1, 2, 4]
    # x = GlobalAveragePooling3D()(drop)
    # spp = SpatialPyramidPooling(pooling_regions)(drop)

    flatten = Flatten()(drop)

    dense2 = Dense(1024, use_bias=False, kernel_regularizer=l2(weight_decay))(flatten)

    x_dense = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal', activation='softmax')(dense2)

    return x_dense


def _handle_dim_ordering():
    # TODO 处理维度
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def ResneXt_IN(input_shape=None, cardinality=8, weight_decay=5e-4, classes=None):
    # TODO 主函数
    # model = ResneXt_IN((1, 11, 11, 200), cardinality=8, classes=16)

    # 判断底层，tf，channel在最后
    _handle_dim_ordering()

    if len(input_shape) != 4:
        raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

    print('original input shape:', input_shape)
    # orignal input shape（1，11，11，200）

    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
    print('change input shape:', input_shape)

    # TODO 数据输入
    input = Input(shape=input_shape)

    # TODO 网络搭建
    x_dense = __create_res_next(classes, input, cardinality, weight_decay)

    model = Model(input, x_dense, name='resnext_IN')

    # feature_model = Model(input, outputs=model.get_layer('dense_1').output)

    return model


def main():
    # TODO 程序入口
    # TODO 可变参数3，cardinality
    model = ResneXt_IN((1, 11, 11, 200),  cardinality=8, classes=16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    plot_model(model, show_shapes=True, to_file='./model_ResNeXt.png')


if __name__ == '__main__':
    main()

# 736,064
