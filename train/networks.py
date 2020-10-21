import train.flows as fnn
import torch
from torch.nn import functional as F
from torch.nn import init
from nflows import flows, transforms
from nflows import distributions as distributions_
from nflows.nn import nets
from sbi.utils.sbiutils import standardizing_net, standardizing_transform
from sbi.utils.torchutils import create_alternating_binary_mask

def likelihoodNetwork(args, training_theta=None, training_x=None):
    modules = []
    if args.likelihoodFlowType == 'nvp':
        mask = torch.arange(0, args.xDim) % 2
        mask = mask.to(args.device).float()

        for _ in range(args.likelihoodNumBlocks):
            modules += [
                fnn.CouplingLayer(
                    args.xDim, 20, mask, num_cond_inputs=args.thetaDim,
                    s_act='tanh', t_act='relu'),
                # fnn.BatchNormFlow(self.x_dim),
                fnn.Shuffle(args.xDim)
            ]
        netLikelihood = fnn.FlowSequential(*modules).to(args.device)

    elif args.likelihoodFlowType == 'maf':
        for _ in range(args.likelihoodNumBlocks):
            modules += [
                fnn.MADE(args.xDim, 50, num_cond_inputs=args.thetaDim, act='relu'),
                fnn.Reverse(args.xDim)
            ]
        netLikelihood = fnn.FlowSequential(*modules).to(args.device)

    elif args.likelihoodFlowType == 'nflow_maf':
        netLikelihood = neural_net_maf(args, args.likelihoodHiddenDim, args.likelihoodNumBlocks, args.xDim, args.thetaDim, training_x, training_theta).to(args.device)

    elif args.likelihoodFlowType == 'nsf':
        netLikelihood = neural_net_nsf(args, args.likelihoodHiddenDim, args.likelihoodNumBlocks, args.likelihoodNumBin,
                                       args.xDim, args.thetaDim, training_x, training_theta, tail=args.nsfTailBound).to(args.device)

    elif args.likelihoodFlowType == 'iaf':
        for _ in range(args.likelihoodNumBlocks):
            modules += [
                fnn.MADE_IAF(args.xDim, 50, num_cond_inputs=args.thetaDim, act='relu'),
                fnn.Reverse(args.xDim)
            ]
        netLikelihood = fnn.FlowSequential(*modules).to(args.device)

    elif args.likelihoodFlowType == 'naf':
        from pyro.nn import AutoRegressiveNN
        from pyro.nn import ConditionalAutoRegressiveNN
        import pyro.distributions as dist
        import pyro.distributions.transforms as T
        import pyro
        baseDistributionLikelihood = dist.Normal(torch.zeros(args.thetaDim),
                                                      torch.ones(args.thetaDim))
        arnLikelihood = ConditionalAutoRegressiveNN(args.thetaDim, args.xDim,
                                                         [40, 40, 40, 40, 40],
                                                         param_dims=[16] * 3)
        nafLikelihood = T.ConditionalNeuralAutoregressive(arnLikelihood, hidden_units=16)
        pyro.module("nafLikelihood", nafLikelihood)
        netLikelihood = dist.ConditionalTransformedDistribution(baseDistributionLikelihood, [nafLikelihood]).to(args.device)

    elif args.likelihoodFlowType == 'bnaf':
        import train.bnaf as bnaf
        for f in range(args.likelihoodNumBlocks):
            layers = []
            for _ in range(1 - 1):
                layers.append(bnaf.MaskedWeight(50,
                                                50, dim=args.xDim,
                                                context_features=args.thetaDim))
                layers.append(bnaf.Tanh())

            modules.append(
                bnaf.BNAF(*([bnaf.MaskedWeight(args.xDim, 50, dim=args.xDim,
                                               context_features=args.thetaDim), bnaf.Tanh()] + \
                            layers + \
                            [bnaf.MaskedWeight(50, args.xDim, dim=args.xDim,
                                               context_features=args.thetaDim)]), \
                          res='gated' if f < args.likelihoodNumBlocks - 1 else None
                          )
            )

            if f < args.likelihoodNumBlocks - 1:
                modules.append(bnaf.Permutation(args.xDim, 'flip'))
        netLikelihood = bnaf.Sequential(*modules).to(args.device)

    elif args.likelihoodFlowType == 'umnn':
        import umnn.models.UMNN.UMNNMAFFlow as umnn
        netLikelihood = umnn.UMNNMAFFlow_(nb_flow=1, nb_in=args.xDim,
                                                hidden_derivative=[40, 40, 40, 40],
                                                hidden_embedding=[40, 40, 40, 40],
                                                embedding_s=10, nb_steps=50,
                                                cond_in=args.thetaDim, device=args.device).to(args.device)

    #if args.likelihoodFlowType == 'nflow_maf':
    #    optimizer = None
    if args.likelihoodFlowType != 'umnn':
        optimizer = torch.optim.Adam(netLikelihood.parameters(), lr=args.lrLikelihood, betas=(0.5, 0.999), weight_decay=1e-5)
    elif args.likelihoodFlowType == 'umnn':
        optimizer = torch.optim.Adam(netLikelihood.parameters(), lr=args.lrLikelihood)#, weight_decay=1e-5)

    return netLikelihood, optimizer

def neural_net_maf(args, hidden_features, num_blocks, xDim, thetaDim, batch_x=None, batch_theta=None):
    if batch_x != None:
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=batch_x.shape[1],
                            hidden_features=hidden_features,
                            context_features=batch_theta.shape[1],
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.1,
                            use_batch_norm=True,
                            device = args.device
                        ),
                        transforms.RandomPermutation(features=batch_x.shape[1], device=args.device),
                    ]
                )
                for _ in range(num_blocks)
            ]
        ).to(args.device)
    else:
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=xDim,
                            hidden_features=hidden_features,
                            context_features=thetaDim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.1,
                            use_batch_norm=True,
                            device = args.device
                        ),
                        transforms.RandomPermutation(features=xDim, device=args.device),
                    ]
                )
                for _ in range(num_blocks)
            ]
        ).to(args.device)

    if batch_theta != None:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])
        embedding_net = torch.nn.Sequential(standardizing_net(batch_theta), torch.nn.Identity())
    if batch_x != None:
        distribution = distributions_.StandardNormal((batch_x.shape[1],), args.device)
    else:
        distribution = distributions_.StandardNormal((xDim,), args.device)
    if batch_theta != None:
        neural_net = flows.Flow(transform, distribution, embedding_net).to(args.device)
    else:
        neural_net = flows.Flow(transform, distribution).to(args.device)

    return neural_net

def neural_net_nsf(args, hidden_features, num_blocks, num_bins, xDim, thetaDim, batch_x=None, batch_theta=None, tail=3.) -> torch.nn.Module:
    """Builds NSF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.PiecewiseRationalQuadraticCouplingTransform(
                        mask=create_alternating_binary_mask(
                            features=xDim, even=(i % 2 == 0)
                        ).to(args.device),
                        transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=hidden_features,
                            context_features=thetaDim,
                            num_blocks=2,
                            activation=torch.relu,
                            dropout_probability=0.,
                            use_batch_norm=False,
                        ),
                        num_bins=num_bins,
                        tails="linear",
                        tail_bound=tail,
                        apply_unconditional_transform=False,
                    ),
                    transforms.LULinear(xDim, identity_init=True),
                ]
            )
            for i in range(num_blocks)
        ]
    ).to(args.device)

    if batch_theta != None:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])
        embedding_net = torch.nn.Sequential(standardizing_net(batch_theta), torch.nn.Identity())
        distribution = distributions_.StandardNormal((xDim,), args.device)
        neural_net = flows.Flow(transform, distribution, embedding_net).to(args.device)
    else:
        distribution = distributions_.StandardNormal((xDim,), args.device)
        neural_net = flows.Flow(transform, distribution).to(args.device)

    return neural_net

class ResidualNet(torch.nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = torch.nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = torch.nn.Linear(in_features, hidden_features)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = torch.nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

class ResidualBlock(torch.nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = torch.nn.Linear(context_features, features)
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = torch.nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)








def discriminatorNetwork(args, training_theta=None, training_x=None):
    netDiscriminator = build_MLP_classifier(args.xDim, args.thetaDim, 256, args.device).to(args.device)
    optimizer = torch.optim.Adam(netDiscriminator.parameters(), lr=args.lrLikelihood, betas=(0.5, 0.999), weight_decay=1e-5)
    return netDiscriminator, optimizer

def build_MLP_classifier(xDim, thetaDim, hidden_features, device='cpu') -> torch.nn.Module:
    hidden_features = [hidden_features, hidden_features, hidden_features]
    neural_net = torch.nn.Sequential(nets.MLP(in_shape=(xDim + thetaDim,), out_shape=(1,), hidden_sizes=hidden_features)).to(device)
    return neural_net

def build_resnet_classifier(xDim, thetaDim,
    batch_x = None,
    batch_y = None,
    z_score_x = True,
    z_score_y = True,
    hidden_features = 50,
    embedding_net_x = torch.nn.Identity(),
    embedding_net_y = torch.nn.Identity(),
        device: str = 'cpu'
) -> torch.nn.Module:
    """Builds ResNet classifier.

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Neural network.
    """

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = xDim
    y_numel = thetaDim

    neural_net = nets.ResidualNet(
        in_features=x_numel + y_numel,
        out_features=1,
        hidden_features=hidden_features,
        context_features=None,
        num_blocks=2,
        activation=torch.relu,
        dropout_probability=0.1,
        use_batch_norm=False,
        device=device
    )

    input_layer = build_input_layer(xDim, thetaDim,
        batch_x, batch_y, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    neural_net = torch.nn.Sequential(input_layer, neural_net).to(device)

    return neural_net

def build_input_layer(xDim, thetaDim,
    batch_x = None,
    batch_y = None,
    z_score_x = True,
    z_score_y = True,
    embedding_net_x = torch.nn.Identity(),
    embedding_net_y = torch.nn.Identity(),
) -> torch.nn.Module:
    """Builds input layer for classifiers that optionally z-scores.

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Input layer that optionally z-scores.
    """
    if batch_x != None:
        embedding_net_x = torch.nn.Sequential(standardizing_net(batch_x), embedding_net_x)

    if batch_y != None:
        embedding_net_y = torch.nn.Sequential(standardizing_net(batch_y), embedding_net_y)

    input_layer = StandardizeInputs(
        embedding_net_x, embedding_net_y, dim_x=xDim, dim_y=thetaDim
    )

    return input_layer

class StandardizeInputs(torch.nn.Module):
    def __init__(self, embedding_net_x, embedding_net_y, dim_x, dim_y):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    def forward(self, t):
        out = torch.cat(
            [
                self.embedding_net_x(t[:, : self.dim_x]),
                self.embedding_net_y(t[:, self.dim_x : self.dim_x + self.dim_y]),
            ],
            dim=1,
        )
        return out




























def posteriorNetwork(args, simulation, training_theta=None, training_x=None):
    if training_theta != None:
        modules = []
        if args.posteriorFlowType == 'maf':
            for _ in range(args.posteriorNumBlocks):
                modules += [
                    fnn.MADE(args.thetaDim, 50, act='relu'),
                    fnn.Reverse(args.thetaDim)
                ]
            netPosterior = fnn.FlowSequential(*modules).to(args.device)

        elif args.posteriorFlowType == 'nflow_maf':
            netPosterior = neural_net_maf(args, args.posteriorHiddenDim, args.posteriorNumBlocks, args.thetaDim, None, None, None).to(args.device)

        elif args.posteriorFlowType == 'nsf':
            bounds = torch.max(torch.max(torch.abs(simulation.max)),torch.max(torch.abs(simulation.min))).item()
            netPosterior = neural_net_nsf(args, args.posteriorHiddenDim, args.posteriorNumBlocks, args.posteriorNumBin, args.thetaDim, None, batch_x=None, batch_theta=None, tail=bounds).to(args.device)

        elif args.posteriorFlowType == 'bnaf':
            import bnaf.bnaf as bnaf
            for f in range(args.posteriorNumBlocks):
                layers = []
                for _ in range(1 - 1):
                    layers.append(bnaf.MaskedWeight(50,
                                                    50, dim=args.thetaDim,
                                                    context_features=None))
                    layers.append(bnaf.Tanh())

                modules.append(
                    bnaf.BNAF(*([bnaf.MaskedWeight(args.thetaDim, 50, dim=args.thetaDim,
                                                   context_features=None), bnaf.Tanh()] + \
                                layers + \
                                [bnaf.MaskedWeight(50, args.thetaDim, dim=args.thetaDim,
                                                   context_features=None)]), \
                              res='gated' if f < args.posteriorNumBlocks - 1 else None
                              )
                )

                if f < args.posteriorNumBlocks - 1:
                    modules.append(bnaf.Permutation(args.thetaDim, 'flip'))
            netPosterior = bnaf.Sequential(*modules).to(args.device)

        elif args.posteriorFlowType == 'umnn':
            import umnn.models.UMNN.UMNNMAFFlow as umnn
            netPosterior = umnn.UMNNMAFFlow_(nb_flow=2, nb_in=args.thetaDim,
                               hidden_derivative=[100, 100, 100, 100],
                               hidden_embedding=[100, 100, 100, 100],
                               embedding_s=10, nb_steps=20, device=args.device,
                               sigmoid=True).to(args.device)

        optimizer = torch.optim.Adam(netPosterior.parameters(), lr=args.lrPosterior, betas=(0.5, 0.999), weight_decay=1e-5)

    else:
        return -1,-1
    #print("posterior : ", netPosterior, optimizer)
    return netPosterior, optimizer