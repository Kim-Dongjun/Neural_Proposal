import torch
import numpy as np

def true_parameter(args):

    if args.simulation == 'shubert':
        return torch.zeros(args.thetaDim).reshape(1,-1).to(args.device)

    elif args.simulation == 'SLCP-16':
        if args.thetaDim != 2:
            return torch.Tensor([[1.5, -2.0, -1., -0.9, 0.6]]).to(args.device)
        else:
            return torch.Tensor([[1.5, -2.9]]).to(args.device)

    elif args.simulation == 'SLCP-256':
        return torch.Tensor([[1.5,2.0,1.3,1.2,1.8,2.5,1.6,1.1]]).to(args.device)

    elif args.simulation == 'mg1':
        return torch.Tensor([[1, 4, 0.2]]).to(args.device)

    elif args.simulation == 'CLV':
        if args.thetaDim == 2:
            return torch.Tensor([[1.52, 0.]]).to(args.device)
        elif args.thetaDim == 3:
            return torch.Tensor([[1.52, 0., 0.51]]).to(args.device)
        elif args.thetaDim == 4:
            return torch.Tensor([[1.52,0.,1.21,0.51]]).to(args.device)
        elif args.thetaDim == 8:
            return torch.Tensor([[1.52, 0., 0.44, 1.36, 2.33, 0., 1.21, 0.51]]).to(args.device)