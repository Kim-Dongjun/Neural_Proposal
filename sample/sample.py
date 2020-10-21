import torch
import time
import sample.mcmc as mcmc
import train.converterNetLikelihoodToNetLikelihoodToEvidenceRatio as converter
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from sbi.types import ScalarFloat
from sbi import utils as utils
from torch import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
from sample.slice import Slice
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
import math
#import tensorflow.compat.v2 as tf
#import tensorflow_probability as tfp
import os
import time

def sample(args, num_samples, netLikelihood, netPosterior, observation, prior, simulator, init=False, mcmc=True, numChains=1, parallel=False, SNLE=True):
    current_time = time.time()
    if observation.get_device() == -1:
        device = 'cpu'
    else:
        device = args.device
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(observation.get_device())
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #tf.config.experimental.set_visible_devices(gpus[observation.get_device()], 'GPU')
        #tf.config.experimental.set_memory_growth(gpus[observation.get_device()], True)
    if not SNLE:
        netLikelihood = converter.netRatio()(netLikelihood)
    if init:
        thetas = torch.Tensor(prior.gen(num_samples))
    else:
        if mcmc:
            mcmc_method=None
            if args.samplerType == 'MHMultiChainsSampler':
                sampler = MHUniformMultiChainsSampler
            elif args.samplerType == 'MHGaussianMultiChainsSampler':
                sampler = MHGaussianMultiChainsSampler
            elif args.samplerType == 'sliceSampler':
                sampler = sliceSampler
            elif args.samplerType == 'sbiSliceSampler' and parallel == False:
                sampler = sample_from_sbi
                mcmc_method = 'slice'
            elif args.samplerType == 'sbiSliceSampler' and parallel == True:
                sampler = MHGaussianMultiChainsSampler
            elif args.samplerType == 'sbiHamiltonianSampler':
                sampler = sample_from_sbi
                mcmc_method = 'hmc'
            elif args.samplerType == 'sbiNUTSSampler':
                sampler = sample_from_sbi
                mcmc_method = 'nuts'
            elif args.samplerType == 'tfSliceSampler':
                sampler = sample_from_tf
                mcmc_method = 'slice'
            elif args.samplerType == 'tfMHSampler':
                sampler = sample_from_tf
                mcmc_method = 'Metropolis-Hastings'

            if mcmc_method == None:
                thetas = torch.cat((sampler(args, int(args.samplerExploitationRatio * num_samples), netLikelihood, observation, prior, simulator, mcmc_method, num_chains=numChains).detach(),
                                    torch.Tensor(prior.gen(num_samples
                                                                - int(args.samplerExploitationRatio * num_samples))).to(
                                        device))).detach()
            else:
                thetas = torch.cat((sampler(args, netLikelihood, prior, int(args.samplerExploitationRatio * num_samples), observation, mcmc_method,
                                            num_chains=numChains, warmup_steps=args.burnInMCMC, thin=args.thinning).detach(),
                                    torch.Tensor(prior.gen(num_samples
                                                                - int(args.samplerExploitationRatio * num_samples))).to(
                                        device))).detach()

        else:
            if args.posteriorInferenceMethod == 'rkl':
                dim = args.thetaDim

            noise = torch.randn([int(args.samplerExploitationRatio * num_samples), dim]).to(args.device)
            #print("netPosterior : ", netPosterior)
            try:
                thetas = netPosterior.sampling(noise).cpu().detach().to(device)
            except:
                thetas = netPosterior.sample(int(args.samplerExploitationRatio * num_samples)).cpu().detach().to(device)
            thetas = torch.cat((thetas, torch.Tensor(simulator.min.cpu().detach().numpy() + (simulator.max - simulator.min).cpu().detach().numpy() * prior.gen(num_samples
                    - int(args.samplerExploitationRatio * num_samples))).to(device))).detach()

    print("sampling time : ", time.time() - current_time)
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
            #print("itr : ", itr)
        #print(str(itr) + " Metropolis-Hastings Time : ", time.time() - current)
    print("Metropolis-Hastings Time : ", time.time() - current)
    return simulator.min + (simulator.max - simulator.min) * mcmc_samples[:n_sample]

def MHGaussianMultiChainsSampler(args, n_sample, netLikelihood, observation, prior=None, simulator=None, mcmc_method=None, num_chains=None, thin=10, proposal_std=0.1):
    if observation.shape[0] == args.xDim:
        observation = observation.repeat(num_chains, 1)
    elif observation.shape[0] != args.xDim:
        observation = observation.repeat(1, num_chains // observation.shape[0]).reshape(-1,args.xDim)

    current = time.time()
    thetas = torch.rand((num_chains, args.thetaDim)).to(args.device)
    mcmc_samples = torch.Tensor([]).to(args.device)
    proposal_std = torch.Tensor([[proposal_std]]).repeat(num_chains, args.thetaDim).to(args.device)
    for itr in range(args.burnInMCMC + thin * math.ceil(n_sample / num_chains)):
        try:
            thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, args.thetaDim)).to(args.device)
            rand = torch.rand(num_chains).to(args.device).reshape(-1)
            mask = (torch.exp(
                torch.min(netLikelihood.log_prob(x=observation, theta=simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)
                            + torch.Tensor(prior.eval(simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(args.device)
                          - netLikelihood.log_probs(x=observation, theta=simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)
                          - torch.Tensor(prior.eval(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(args.device),
                          torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
        except:
            try:
                thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, args.thetaDim)).to(args.device)
                rand = torch.rand(num_chains).to(args.device).reshape(-1)
                mask = (torch.exp(
                    torch.min(netLikelihood.log_prob(context=simulator.min + (simulator.max - simulator.min) * thetas_intermediate, inputs=observation).reshape(-1)
                              + torch.Tensor(prior.eval(simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(-1)).to(args.device)
                              - netLikelihood.log_prob(context=simulator.min + (simulator.max - simulator.min) * thetas.to(args.device), inputs=observation).reshape(-1)
                              - torch.Tensor(prior.eval(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(args.device),
                              torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
            except:
                thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, args.thetaDim)).to(args.device)
                rand = torch.rand(num_chains).to(args.device).reshape(-1)
                mask = (torch.exp(
                    torch.min(netLikelihood(
                        context=simulator.min + (simulator.max - simulator.min) * thetas_intermediate,
                        inputs=observation).reshape(-1)
                              + torch.Tensor(
                        prior.eval(simulator.min + (simulator.max - simulator.min) * thetas_intermediate).reshape(
                            -1)).to(args.device)
                              - netLikelihood(
                        context=simulator.min + (simulator.max - simulator.min) * thetas.to(args.device),
                        inputs=observation).reshape(-1)
                              - torch.Tensor(
                        prior.eval(simulator.min + (simulator.max - simulator.min) * thetas).reshape(-1)).to(
                        args.device),
                              torch.Tensor([0.] * num_chains).to(args.device))) > rand).float().reshape(-1, 1)
        if itr == 0:
            masks = mask.reshape(1, -1, 1)
        else:
            masks = torch.cat((masks, mask.reshape(1, -1, 1)))
        if itr % thin == 0:
            bool = (torch.sum(masks[-100:,:,:],0) / 100 > 0.234).float()
            proposal_std = (1.1 * bool + 0.9 * (1 - bool)).repeat(1,args.thetaDim).to(args.device) * proposal_std
        #print("proposal std : ", proposal_std)
        thetas = thetas_intermediate * mask + thetas.to(args.device) * (1 - mask)
        if max(itr - args.burnInMCMC, 0) % thin == thin - 1:
            mcmc_samples = torch.cat((mcmc_samples, thetas))
    print("Metropolis-Hastings Time : ", time.time() - current)
    return simulator.min + (simulator.max - simulator.min) * mcmc_samples[:n_sample]

def sliceSampler(args, n_sample, netLikelihood, observation, prior=None, simulator=None, mcmc_method=None, num_chains=None):
    if args.likelihoodFlowType not in ['nflow_maf', 'nsf']:
        log_posterior = lambda t: netLikelihood.log_probs(observation.reshape(1,-1),torch.Tensor(t).to(args.device).reshape(1,-1))\
                                  + prior.eval(t)
        sliceSampler = mcmc.SliceSampler(prior.gen(), log_posterior, thin=10)
        thetas = torch.Tensor(sliceSampler.gen(n_sample)).to(args.device)
    else:
        device = 'cpu'
        log_posterior = lambda t: netLikelihood.log_prob(observation.reshape(1, -1).cpu(),
                                                          torch.Tensor(t).to(device).reshape(1, -1)) \
                                  + prior.eval(t)
        sliceSampler = mcmc.SliceSampler(prior.gen(), log_posterior, thin=10)
        thetas = torch.Tensor(sliceSampler.gen(n_sample)).to(device)
    return thetas

def sample_from_sbi(args, netLikelihood, prior,
    num_samples: int,
    x,
    mcmc_method: str = "slice",
    thin: int = 10,
    warmup_steps: int = 20,
    num_chains = 25,
    init_strategy: str = "prior",
    init_strategy_num_candidates: int = 10000,
    show_progress_bars: bool = True,
    _mcmc_init_params = None
):
    r"""
    Return MCMC samples from posterior $p(\theta|x)$.

    This function is used in any case by SNLE and SNRE, but can also be used by SNPE
    in order to deal with strong leakage. Depending on the inference method, a
    different potential function for the MCMC sampler is required.

    Args:
        num_samples: Desired number of samples.
        x: Conditioning context for posterior $p(\theta|x)$.
        mcmc_method: Sampling method. Currently defaults to `slice_np` for a custom
            numpy implementation of slice sampling; select `hmc`, `nuts` or `slice`
            for Pyro-based sampling.
        thin: Thinning factor for the chain, e.g. for `thin=3` only every third
            sample will be returned, until a total of `num_samples`.
        warmup_steps: Initial number of samples to discard.
        num_chains: Whether to sample in parallel. If None, use all but one CPU.
        init_strategy: Initialisation strategy for chains; `prior` will draw init
            locations from prior, whereas `sir` will use Sequential-Importance-
            Resampling using `init_strategy_num_candidates` to find init
            locations.
        init_strategy_num_candidates: Number of candidate init locations
            when `init_strategy` is `sir`.
        show_progress_bars: Whether to show a progressbar during sampling.

    Returns:
        Tensor of shape (num_samples, shape_of_single_theta).
    """
    # Find init points depending on `init_strategy` if no init is set
    print("num samples : ", num_samples)
    if _mcmc_init_params is None:
        if init_strategy == "prior":
            _mcmc_init_params = torch.Tensor(prior.gen(num_chains)).to(args.device).detach().reshape(-1,prior.lower.shape[0])

    potential_function = PotentialFunctionProvider()(args, prior, netLikelihood, x, mcmc_method)

    track_gradients = mcmc_method != "slice" and mcmc_method != "slice_np"
    with torch.set_grad_enabled(track_gradients):
        if mcmc_method == "slice_np":
            samples = _slice_np_mcmc(netLikelihood, _mcmc_init_params, prior,
                num_samples, potential_function, thin, warmup_steps,
            )
        elif mcmc_method in ("hmc", "nuts", "slice"):
            samples = _pyro_mcmc(args, netLikelihood, _mcmc_init_params, prior,
                num_samples=num_samples,
                potential_function=potential_function,
                mcmc_method=mcmc_method,
                thin=thin,
                warmup_steps=warmup_steps,
                num_chains=num_chains,
                show_progress_bars=show_progress_bars,
            ).detach()
        else:
            raise NameError

    return samples

def sample_from_tf(args, netLikelihood, prior,
    num_samples: int,
    x,
    mcmc_method: str = "slice",
    thin: int = 10,
    warmup_steps: int = 20,
    num_chains = 25,
    init_strategy: str = "prior",
    init_strategy_num_candidates: int = 10000,
    show_progress_bars: bool = True,
    _mcmc_init_params = None
):
    print("num samples : ", num_samples)
    if _mcmc_init_params is None:
        if init_strategy == "prior":
            _mcmc_init_params = torch.Tensor(prior.gen(num_chains)).to(args.device).detach().reshape(-1, prior.lower.shape[0])

    potential_function = PotentialFunctionProvider_tf()(args, prior, netLikelihood, x, mcmc_method)

    samples = tf_mcmc(netLikelihood, prior,
                         num_samples=num_samples,
                         potential_function=potential_function,
                         mcmc_method=mcmc_method,
                         thin=thin,
                         warmup_steps=warmup_steps,
                         num_chains=num_chains,
                         )

    return samples.to(args.device).detach()

class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __call__(
        self, args, prior, likelihood_nn: torch.nn.Module, x: torch.Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.args = args
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        # Notice opposite sign to pyro potential.
        return log_likelihood + self.prior.eval(theta.cpu().detach().numpy())

    def pyro_potential(self, theta):
        r"""Return posterior log probability of parameters $p(\theta|x)$.

         Args:
            theta: Parameters $\theta$. The tensor's shape will be
                (1, shape_of_single_theta) if running a single chain or just
                (shape_of_single_theta) for multiple chains.

        Returns:
            The potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
        """

        theta = next(iter(theta.values()))

        try:
            log_likelihood = self.likelihood_nn.log_prob(
                inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
            )
        except:
            log_likelihood = self.likelihood_nn(inputs=self.x.reshape(1,-1), context=theta.reshape(1,-1))

        return -(log_likelihood + self.prior.eval(theta.cpu().detach().numpy()).item())

class PotentialFunctionProvider_tf:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __call__(
            self, args, prior, likelihood_nn: torch.nn.Module, x: torch.Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.args = args
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        return self.tf_potential

    def tf_potential(self, theta):
        theta = torch.Tensor(theta.numpy()).to(self.args.device)

        try:
            log_likelihood = self.likelihood_nn.log_prob(
                inputs=self.x.reshape(1,-1), context=theta
            )
        except:
            try:
                log_likelihood = self.likelihood_nn.log_prob(
                    inputs=self.x.reshape(1, -1).repeat(theta.shape[0],1), context=theta
                )
            except:
                log_likelihood = self.likelihood_nn.log_prob(
                    x=self.x.reshape(1, -1), theta=theta
                )

        return tf.convert_to_tensor(log_likelihood.cpu().detach() + self.prior.eval(theta.cpu().detach().numpy()), dtype=tf.float32)

def _slice_np_mcmc(netLikelihood, _mcmc_init_params, prior,
    num_samples: int,
    potential_function: Callable,
    thin: int,
    warmup_steps: int,
) -> torch.Tensor:
    """
    Custom implementation of slice sampling using Numpy.

    Args:
        num_samples: Desired number of samples.
        potential_function: A callable **class**.
        thin: Thinning (subsampling) factor.
        warmup_steps: Initial number of samples to discard.

    Returns: Tensor of shape (num_samples, shape_of_single_theta).
    """
    # Go into eval mode for evaluating during sampling
    netLikelihood.eval()

    num_chains = _mcmc_init_params.shape[0]
    dim_samples = _mcmc_init_params.shape[1]

    all_samples = []
    for c in range(num_chains):
        posterior_sampler = mcmc.SliceSampler(
            utils.tensor2numpy(_mcmc_init_params[c, :]).reshape(-1),
            lp_f=potential_function,
            thin=thin,
        )
        if warmup_steps > 0:
            posterior_sampler.gen(int(warmup_steps))
        all_samples.append(posterior_sampler.gen(int(num_samples / num_chains)))
    all_samples = np.stack(all_samples).astype(np.float32)

    samples = torch.from_numpy(all_samples)  # chains x samples x dim

    # Final sample will be next init location
    _mcmc_init_params = samples[:, -1, :].reshape(num_chains, dim_samples)

    samples = samples.reshape(-1, dim_samples)[:num_samples, :]
    assert samples.shape[0] == num_samples

    # Back to training mode
    netLikelihood.train(True)

    return samples.type(torch.float32)

def tf_mcmc(netLikelihood, prior, num_samples, potential_function, mcmc_method = 'slice', thin = 10, warmup_steps = 200, num_chains = 1):

    tf.enable_v2_behavior()
    if mcmc_method == 'slice':
        samples = torch.Tensor(tfp.mcmc.sample_chain(num_results = num_samples,
                                        current_state = tf.random.uniform([num_chains, prior.lower.shape[0]],
                                                                          name='init_weights', minval=prior.lower, maxval=prior.upper),
                                        kernel=tfp.mcmc.SliceSampler(
                                            potential_function,
                                            step_size=1.0,
                                            max_doublings=5
                                        ),
                                        num_burnin_steps = warmup_steps,
                                        trace_fn = None
                                        ).numpy()).reshape(-1,prior.lower.shape[0])
    elif mcmc_method == 'Metropolis-Hastings':
        samples = torch.Tensor(tfp.mcmc.sample_chain(num_results=math.ceil(num_samples/num_chains),
                                                     current_state=tf.random.uniform([num_chains, prior.lower.shape[0]],
                                                                            name='init_weights', minval=prior.lower, maxval=prior.upper),
                                                     kernel=tfp.mcmc.RandomWalkMetropolis(
                                                         potential_function),
                                                     num_burnin_steps=warmup_steps,
                                                     num_steps_between_results=9,
                                                     trace_fn=None
                                                     ).numpy())
    elif mcmc_method == 'Hamiltonian':
        samples = torch.Tensor(tfp.mcmc.sample_chain(num_results=math.ceil(num_samples / num_chains),
                                                     current_state=tf.random.uniform([num_chains, prior.lower.shape[0]],
                                                                                     name='init_weights',
                                                                                     minval=prior.lower,
                                                                                     maxval=prior.upper),
                                                     #kernel=tfp.mcmc.SimpleStepSizeAdaptation(
                                                     #       tfp.mcmc.HamiltonianMonteCarlo(
                                                     #           target_log_prob_fn=potential_function,
                                                     #           num_leapfrog_steps=2,
                                                     #           step_size=1.,
                                                     #           state_gradients_are_stopped=True,
                                                     #       ),
                                                            # Adapt for the entirety of the trajectory.
                                                     #       num_adaptation_steps=int(warmup_steps * 0.5)),
                                                     kernel=tfp.mcmc.HamiltonianMonteCarlo(
                                                             target_log_prob_fn=potential_function,
                                                             num_leapfrog_steps=2,
                                                             step_size=1.,
                                                             state_gradients_are_stopped=True,
                                                         ),
                                                     num_burnin_steps=warmup_steps,
                                                     num_steps_between_results=9,
                                                     trace_fn=None
                                                     ).numpy())
    return samples.reshape(-1, prior.lower.shape[0])

def _pyro_mcmc(args, netLikelihood, _mcmc_init_params, prior,
    num_samples: int,
    potential_function: Callable,
    mcmc_method: str = "slice",
    thin: int = 10,
    warmup_steps: int = 200,
    num_chains: Optional[int] = 1,
    show_progress_bars: bool = True,
):
    r"""Return samples obtained using Pyro HMC, NUTS or slice kernels.

    Args:
        num_samples: Desired number of samples.
        potential_function: A callable **class**. A class, but not a function,
            is picklable for Pyro MCMC to use it across chains in parallel,
            even when the potential function requires evaluating a neural network.
        mcmc_method: One of `hmc`, `nuts` or `slice`.
        thin: Thinning (subsampling) factor.
        warmup_steps: Initial number of samples to discard.
        num_chains: Whether to sample in parallel. If None, use all but one CPU.
        show_progress_bars: Whether to show a progressbar during sampling.

    Returns: Tensor of shape (num_samples, shape_of_single_theta).
    """
    num_chains = mp.cpu_count - 1 if num_chains is None else num_chains

    # Go into eval mode for evaluating during sampling
    #netLikelihood.eval()

    kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)

    #if torch.cuda.is_available():
    #    mp_context = 'spawn'
    #else:
    #    mp_context = 'fork'

    sampler = MCMC(
        kernel=kernels[mcmc_method](potential_fn=potential_function),
        num_samples=(thin * num_samples) // num_chains + num_chains,
        warmup_steps=warmup_steps,
        initial_params={"": _mcmc_init_params},
        num_chains=num_chains,
        mp_context='spawn',
        disable_progbar=not show_progress_bars,
    )
    sampler.run()
    samples = next(iter(sampler.get_samples().values())).reshape(
        -1, prior.lower.shape[0]  # len(prior.mean) = dim of theta
    )

    samples = samples[::thin][:num_samples]
    assert samples.shape[0] == num_samples

    # Back to training mode
    #netLikelihood.train(True)

    return samples