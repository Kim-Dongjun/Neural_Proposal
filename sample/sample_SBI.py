import torch
import time
import math

def sample(args, num_samples, netLikelihood, netPosterior, observation, prior, simulator, init=False, mcmc=True, numChains=1, mcmc_parameters={}):
    if observation.get_device() == -1:
        device = 'cpu'
    else:
        device = args.device
    if init:
        thetas = torch.Tensor(prior.sample((num_samples,)))
    else:
        if mcmc:
            mcmc_method = None
            if args.samplerType == 'MHMultiChainsSampler':
                sampler = MHMultiChainsSampler
            elif args.samplerType == 'MHMultiChainsSampler':
                sampler = MHUniformMultiChainsSampler
            elif args.samplerType == 'MHGaussianMultiChainsSampler':
                sampler = MHGaussianMultiChainsSampler
            elif args.samplerType == 'sliceSampler':
                sampler = sliceSampler
            thetas = torch.cat((sampler(args, int(args.samplerExploitationRatio * num_samples), netLikelihood, observation, prior, simulator, mcmc_method, num_chains=numChains).detach(),
                                (prior.sample((num_samples
                                                            - int(args.samplerExploitationRatio * num_samples),))).to(
                                    device))).detach()

        else:
            if args.posteriorInferenceMethod == 'rkl':
                dim = args.thetaDim

            elif args.posteriorInferenceMethod == 'implicit':
                dim = args.posteriorInputDim
            print("sampling ...")
            #noise = torch.randn([int(args.samplerExploitationRatio * num_samples), dim]).to(args.device)
            if mcmc_parameters != {}:
                thetas = netPosterior.sample(sample_shape=(int(args.samplerExploitationRatio * num_samples),), mcmc_parameters=mcmc_parameters, device=args.device).cpu().detach().to(device)
            else:
                try:
                    thetas = netPosterior.sample(sample_shape=(int(args.samplerExploitationRatio * num_samples),), device=args.device).cpu().detach().to(device)
                except:
                    thetas = netPosterior.sample(sample_shape=(int(args.samplerExploitationRatio * num_samples),)).cpu().detach().to(device)
            thetas = torch.cat((thetas, (prior.sample((num_samples
                    - int(args.samplerExploitationRatio * num_samples),))).to(device)))

    return thetas

def MHMultiChainsSampler(args, n_sample, netLikelihood, observation, simulator=None):
    current = time.time()
    thetas = simulator.min + (simulator.max - simulator.min) * torch.rand((n_sample, args.thetaDim)).to(args.device)
    if observation.get_device() == -1:
        device = 'cpu'
    else:
        device = args.device
    for itr in range(args.burnInMCMC):
        thetas_intermediate = simulator.min + (simulator.max - simulator.min) * torch.rand((n_sample, args.thetaDim)).to(args.device)
        rand = torch.rand(n_sample).to(args.device).reshape(-1)
        try:
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(thetas_intermediate, observation).reshape(-1)
                          - netLikelihood.log_prob(thetas, observation).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)
        except:
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(observation, thetas_intermediate).reshape(-1)
                          - netLikelihood.log_prob(observation, thetas).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)

        thetas = thetas_intermediate * mask + thetas.to(device) * (1 - mask)
        #print(str(itr) + " Metropolis-Hastings Time : ", time.time() - current)
    print("Metropolis-Hastings Time : ", time.time() - current)
    return thetas

def MHUniformMultiChainsSampler(args, n_sample, netLikelihood, observation, prior=None, simulator=None, mcmc_method=None, num_chains=None, thin=10):
    if observation.shape[0] == args.xDim:
        observation = observation.repeat(n_sample, 1)
    elif observation.shape[0] != args.xDim:
        observation = observation.repeat(1, n_sample // observation.shape[0]).reshape(-1,args.xDim)

    current = time.time()
    thetas = torch.rand((n_sample, args.thetaDim)).to(args.device)
    mcmc_samples = torch.Tensor([]).to(args.device)
    for itr in range(args.burnInMCMC + thin * math.ceil(n_sample / num_chains)):
        try:
            thetas_intermediate = torch.rand((n_sample, args.thetaDim)).to(args.device)
            rand = torch.rand(n_sample).to(args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_probs(observation, simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                          - netLikelihood.log_probs(observation, simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)
        except:
            thetas_intermediate = torch.rand((n_sample, args.thetaDim)).to(args.device)
            rand = torch.rand(n_sample).to(args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(observation, simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                          - netLikelihood.log_prob(observation, simulator.min + (simulator.max - simulator.min) * thetas.to(args.device)).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)
        thetas = thetas_intermediate * mask + thetas.to(args.device) * (1 - mask)
        if max(itr - args.burnInMCMC, 0) % thin == thin - 1:
            mcmc_samples = torch.cat((mcmc_samples, thetas))
    print("Metropolis-Hastings Time : ", time.time() - current)
    return simulator.min + (simulator.max - simulator.min) * mcmc_samples[:n_sample]

def MHGaussianMultiChainsSampler_(args, n_sample, netLikelihood, observation, prior=None, simulator=None, mcmc_method=None, num_chains=None, thin=10):
    current = time.time()
    thetas = torch.clamp(torch.rand((n_sample, args.thetaDim)).to(args.device), min=0, max=1)
    mcmc_samples = torch.Tensor([]).to(args.device)
    for itr in range(args.burnInMCMC + thin * math.ceil(n_sample / num_chains)):
        try:
            thetas_intermediate = torch.clamp(thetas + 0.1 * torch.randn((n_sample, args.thetaDim)).to(args.device), min=0, max=1)
            rand = torch.rand(n_sample).to(args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_probs(observation, simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                          - netLikelihood.log_probs(observation, simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)
        except:
            thetas_intermediate = torch.clamp(thetas + 0.1 * torch.randn((n_sample, args.thetaDim)).to(args.device), min=0, max=1)
            rand = torch.rand(n_sample).to(args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(observation, simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                          - netLikelihood.log_prob(observation, simulator.min + (simulator.max - simulator.min) * thetas.to(args.device)).reshape(-1),
                          torch.Tensor([0.] * n_sample).to(args.device))) > rand).float().reshape(-1, 1)
        thetas = thetas_intermediate * mask + thetas.to(args.device) * (1 - mask)
        if max(itr - args.burnInMCMC, 0) % thin == thin - 1:
            mcmc_samples = torch.cat((mcmc_samples, thetas))
    print("Metropolis-Hastings Time : ", time.time() - current)
    return simulator.min + (simulator.max - simulator.min) * mcmc_samples[:n_sample]

def MHGaussianMultiChainsSampler(args, n_sample, netLikelihood, observation, prior=None, simulator=None, mcmc_method=None, num_chains=None, thin=10, proposal_std=0.1):
    if args.algorithm != 'SNLE':
        if observation.shape[0] == args.xDim:
            observation = observation.repeat(num_chains, 1)
        elif observation.shape[0] != args.xDim:
            observation = observation.repeat(1, num_chains // observation.shape[0]).reshape(-1,args.xDim)
    else:
        observation = observation.reshape(1,-1)

    current = time.time()
    thetas = torch.rand((num_chains, args.thetaDim)).to(args.device)
    mcmc_samples = torch.Tensor([]).to(args.device)
    proposal_std = torch.Tensor([[proposal_std]]).repeat(num_chains, args.thetaDim).to(args.device)
    for itr in range(args.burnInMCMC + thin * math.ceil(n_sample / num_chains)):
        thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, args.thetaDim)).to(args.device)
        rand = torch.rand(num_chains).to(args.device).reshape(-1)
        if args.algorithm != 'SNLE':
            try:
                mask = (torch.exp(
                    torch.min(netLikelihood.log_prob(x=observation, theta=simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                              + (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(args.device)
                              - netLikelihood.log_prob(x=observation, theta=simulator.min + (simulator.max - simulator.min) * thetas.to(args.device)).reshape(-1)
                              - (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(args.device),
                              torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
            except:
                try:
                    mask = (torch.exp(
                        torch.min(netLikelihood.log_prob(observation, simulator.min + (
                                    simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                                  + (prior.log_prob(
                            simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(args.device)
                                  - netLikelihood.log_prob(observation,
                                                           simulator.min + (simulator.max - simulator.min) * thetas.to(
                                                               args.device)).reshape(-1)
                                  - (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(
                            args.device),
                                  torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
                except:
                    mask = (torch.exp(
                        torch.min(netLikelihood.log_prob(inputs=observation, context=simulator.min + (
                                simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                                  + (prior.log_prob(
                            simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(
                            args.device)
                                  - netLikelihood.log_prob(inputs=observation,
                                                           context=simulator.min + (simulator.max - simulator.min) * thetas.to(
                                                               args.device)).reshape(-1)
                                  - (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas).reshape(
                            -1)).to(
                            args.device),
                                  torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
        else:
            try:
                mask = (torch.exp(
                    torch.min(netLikelihood.log_prob(theta=simulator.min + (
                            simulator.max - simulator.min) * thetas_intermediate, x=observation).reshape(-1)
                              + (prior.log_prob(
                        simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(args.device)
                              - netLikelihood.log_prob(theta=simulator.min + (simulator.max - simulator.min) * thetas.to(
                                                           args.device), x=observation).reshape(-1)
                              - (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(
                        args.device),
                              torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
            except:
                mask = (torch.exp(
                    torch.min(netLikelihood.log_prob(simulator.min + (
                            simulator.max - simulator.min) * thetas_intermediate, observation).reshape(-1)
                              + (prior.log_prob(
                        simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(
                        args.device)
                              - netLikelihood.log_prob(
                        simulator.min + (simulator.max - simulator.min) * thetas.to(
                            args.device), observation).reshape(-1)
                              - (prior.log_prob(simulator.min + (simulator.max - simulator.min) * thetas).reshape(
                        -1)).to(
                        args.device),
                              torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
        if itr == 0:
            masks = mask.reshape(1, -1, 1)
        else:
            masks = torch.cat((masks, mask.reshape(1, -1, 1)))
        if itr % thin == 0:
            bool = (torch.sum(masks[-100:,:,:],0) / 100 > 0.234).float()
            proposal_std = (1.1 * bool + 0.9 * (1 - bool)).repeat(1,args.thetaDim).to(args.device) * proposal_std
        '''if num_chains < 100:
            if itr == 0:
                masks = mask.reshape(1,-1,1)
            else:
                masks = torch.cat((masks, mask.reshape(1,-1,1)))
            if torch.sum(masks.reshape(-1)[-n_sample:]) / n_sample > 0.234:
                proposal_std *= 1.1
            else:
                proposal_std *= 0.9
        else:'''
        '''if num_chains == n_sample:
            #print("!!! : ", n_sample, mask.shape, torch.sum(mask) / num_chains)
            if torch.sum(mask) / num_chains > 0.234:
                proposal_std *= 1.1
            else:
                proposal_std *= 0.9
        if proposal_std > 1:
            proposal_std = 0.01'''
        #print("proposal std : ", proposal_std)
        thetas = thetas_intermediate * mask + thetas * (1 - mask)
        if max(itr - args.burnInMCMC, 0) % thin == thin - 1:
            mcmc_samples = torch.cat((mcmc_samples, thetas))
    print("Metropolis-Hastings Time : ", time.time() - current)
    return simulator.min + (simulator.max - simulator.min) * mcmc_samples[:n_sample]

def sliceSampler(args, n_sample, netLikelihood, observation, prior):
    thetas = netLikelihood.sample(n_sample)
    return thetas