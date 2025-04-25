
    def _initialize_weights(self, active_sd=0.0001, passive_sd=0.1):
        """Initialize a 2D TensorFlow Variable containing normally-distributed
        random weights for an ``N`` mode quantum neural network with ``L`` layers.

        Args:
            modes (int): the number of modes in the quantum neural network
            layers (int): the number of layers in the quantum neural network
            active_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the active parameters
                (displacement, squeezing, and Kerr magnitude)
            passive_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the passive parameters
                (beamsplitter angles and all gate phases)

        Returns:
            tf.Variable[tf.float32]: A TensorFlow Variable of shape
            ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
            row represents the layer parameters for the Lth layer.
        """
        # Number of interferometer parameters:
        num_interfermoter_params = int(self.num_qumodes * (self.num_qumodes - 1)) + max(
            1, self.num_qumodes - 1
        )

        # Create the TensorFlow variables
        int1_weights = (
            torch.randn(self.num_layers, num_interfermoter_params, device=self.device)
            * passive_sd
        )
        s_weights = (
            torch.randn(self.num_layers, self.num_qumodes, device=self.device)
            * active_sd
        )
        int2_weights = (
            torch.randn(self.num_layers, num_interfermoter_params, device=self.device)
            * passive_sd
        )
        dr_weights = (
            torch.randn(self.num_layers, self.num_qumodes, device=self.device)
            * active_sd
        )
        dp_weights = (
            torch.randn(self.num_layers, self.num_qumodes, device=self.device)
            * passive_sd
        )
        k_weights = (
            torch.randn(self.num_layers, self.num_qumodes, device=self.device)
            * active_sd
        )

        weights = torch.cat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights],
            dim=1,
        )

        weights = nn.Parameter(weights, requires_grad=True)

        # print(f"weights: {weights.shape}")
        return weights

    # def data_encoding(self, x):

    #     # for sample in x:
    #     #     print(f"sample: {x[1][0]}")
    #     qml.Squeezing(x[0][0], x[1][0], wires=0)
    #     qml.Squeezing(x[2][0], x[3][0], wires=1)
    #     qml.Squeezing(x[4][0], x[5][0], wires=2)
    #     qml.Squeezing(x[6][0], x[7][0], wires=3)
    #     qml.Squeezing(x[8][0], x[9][0], wires=4)
    #     qml.Squeezing(x[10][0], x[11][0], wires=5)
    #     qml.Squeezing(x[12][0], x[13][0], wires=6)
    #     qml.Squeezing(x[14][0], x[15][0], wires=7)

    #     qml.Beamsplitter(x[16][0], x[17][0], wires=[0, 1])
    #     qml.Beamsplitter(x[18][0], x[19][0], wires=[1, 2])
    #     qml.Beamsplitter(x[20][0], x[21][0], wires=[2, 3])
    #     qml.Beamsplitter(x[22][0], x[23][0], wires=[3, 4])
    #     qml.Beamsplitter(x[24][0], x[25][0], wires=[4, 5])
    #     qml.Beamsplitter(x[26][0], x[27][0], wires=[5, 6])
    #     qml.Beamsplitter(x[28][0], x[29][0], wires=[6, 7])

    #     qml.Rotation(x[30][0], wires=0)
    #     qml.Rotation(x[31][0], wires=1)
    #     qml.Rotation(x[32][0], wires=2)
    #     qml.Rotation(x[33][0], wires=3)
    #     qml.Rotation(x[34][0], wires=4)
    #     qml.Rotation(x[35][0], wires=5)
    #     qml.Rotation(x[36][0], wires=6)
    #     qml.Rotation(x[37][0], wires=7)

    #     qml.Displacement(x[38][0], x[39][0], wires=0)
    #     qml.Displacement(x[40][0], x[41][0], wires=1)
    #     qml.Displacement(x[42][0], x[43][0], wires=2)
    #     qml.Displacement(x[44][0], x[45][0], wires=3)
    #     qml.Displacement(x[46][0], x[47][0], wires=4)
    #     qml.Displacement(x[48][0], x[49][0], wires=5)
    #     qml.Displacement(x[50][0], x[51][0], wires=6)
    #     qml.Displacement(x[52][0], x[53][0], wires=7)

    #     qml.Kerr(x[54][0], wires=0)
    #     qml.Kerr(x[55][0], wires=1)
    #     qml.Kerr(x[56][0], wires=2)
    #     qml.Kerr(x[57][0], wires=3)
    #     qml.Kerr(x[58][0], wires=4)
    #     qml.Kerr(x[59][0], wires=5)
    #     qml.Kerr(x[60][0], wires=6)
    #     qml.Kerr(x[61][0], wires=7)




        # M = int(self.num_qumodes * (self.num_qumodes - 1)) + max(
        #     1, self.num_qumodes - 1
        # )

        # int1 = params[:M]
        # s = params[M : M + self.num_qumodes]
        # int2 = params[M + self.num_qumodes : 2 * M + self.num_qumodes]
        # dr = params[2 * M + self.num_qumodes : 2 * M + 2 * self.num_qumodes]
        # dp = params[2 * M + 2 * self.num_qumodes : 2 * M + 3 * self.num_qumodes]
        # k = params[2 * M + 3 * self.num_qumodes : 2 * M + 4 * self.num_qumodes]

        # begin layer
        # print(f"int2: {int2.shape}")