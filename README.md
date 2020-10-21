# Sequential Likelihood-Free Inference with Implicit Surrogate Proposal

Code for reproducing the experiments in the paper:

>Kim, Dongjun, et al. "Sequential Likelihood-Free Inference with Implicit Surrogate Proposal." arXiv preprint arXiv:2010.07604 (2020).

## Dependencies

```
python: 3.6
pyTorch: 1.5.1
```

## How to reproduce performances

```
The followings are the commands for the experiments.

1) Shubert Simulation

1-1) SMC-ABC
python3.6 SBI.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --algorithm SMC --nsfTailBound 10

1-2) APT
python3.6 SBI.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --algorithm SNPE --nsfTailBound 10

1-3) SNL + Slice sampler with a single chain
python3.6 SNL.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 1

1-4) SNL + Slice sampler with 10 chains
python3.6 SNL.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 10

1-5) SNL + Metropolis-Hastings sampler with a single chain
python3.6 SNL.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 1

1-6) SNL + Metropolis-Hastings sampler with 100 chains
python3.6 SNL.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 100

1-7) AALR + Slice sampler with a single chain
python3.6 AALR.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 1

1-8) AALR + Slice sampler with 10 chains
python3.6 AALR.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 10

1-9) AALR + Metropolis-Hastings sampler with a single chain
python3.6 AALR.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 1

1-10) AALR + Metropolis-Hastings sampler with 10 chains
python3.6 AALR.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod no --device cuda:0 --numChains 100

1-11) SNL + Implicit Surrogate Proposal
python3.6 SNL.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod rkl --device cuda:0

1-12) AALR + Implicit Surrogate Proposal
python3.6 AALR.py --simulation shubert --thetaDim 2 --xDim 2 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 5000 --numRound 10 --numModes 18 --posteriorInferenceMethod rkl --device cuda:0

2) SLCP-16 Simulation

2-1) SMC-ABC
python3.6 SBI.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --algorithm SMC

2-2) APT
python3.6 SBI.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --algorithm SNPE

2-3) SNL + Slice sampler with a single chain
python3.6 SNL.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 1

2-4) SNL + Slice sampler with 10 chains
python3.6 SNL.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 10

2-5) SNL + Metropolis-Hastings sampler with a single chain
python3.6 SNL.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 1

2-6) SNL + Metropolis-Hastings sampler with 100 chains
python3.6 SNL.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 100

2-7) AALR + Slice sampler with a single chain
python3.6 AALR.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 1

2-8) AALR + Slice sampler with 10 chains
python3.6 AALR.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 10

2-9) AALR + Metropolis-Hastings sampler with a single chain
python3.6 AALR.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 1

2-10) AALR + Metropolis-Hastings sampler with 10 chains
python3.6 AALR.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod no --device cuda:0 --numChains 100

2-11) SNL + Implicit Surrogate Proposal
python3.6 SNL.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod rkl --device cuda:0

2-12) AALR + Implicit Surrogate Proposal
python3.6 AALR.py --simulation complexPosterior --thetaDim 5 --xDim 50 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 16 --posteriorInferenceMethod rkl --device cuda:0

3) SLCP-256 Simulation

3-1) SMC-ABC
python3.6 SBI.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --algorithm SMC

3-2) APT
python3.6 SBI.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --algorithm SNPE

3-3) SNL + Slice sampler with a single chain
python3.6 SNL.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --thinning 1

3-4) SNL + Slice sampler with 10 chains
python3.6 SNL.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 10 --thinning 1

3-5) SNL + Metropolis-Hastings sampler with a single chain
python3.6 SNL.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 1

3-6) SNL + Metropolis-Hastings sampler with 100 chains
python3.6 SNL.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 100

3-7) AALR + Slice sampler with a single chain
python3.6 AALR.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 1

3-8) AALR + Slice sampler with 10 chains
python3.6 AALR.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 10

3-9) AALR + Metropolis-Hastings sampler with a single chain
python3.6 AALR.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 1

3-10) AALR + Metropolis-Hastings sampler with 10 chains
python3.6 AALR.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod no --device cuda:0 --numChains 100

3-11) SNL + Implicit Surrogate Proposal
python3.6 SNL.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod rkl --device cuda:0

3-12) AALR + Implicit Surrogate Proposal
python3.6 AALR.py --simulation complexPosterior_ver2 --thetaDim 8 --xDim 40 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS True --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numModes 256 --posteriorInferenceMethod rkl --device cuda:0

4) M/G/1

4-1) SMC-ABC
python3.6 SBI.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numRound 10 --numTime 50 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --algorithm SMC --nsfTailBound 20

4-2) APT
python3.6 SBI.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numRound 10 --numTime 50 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --algorithm SNPE --nsfTailBound 20

4-3) SNL + Slice sampler with a single chain
python3.6 SNL.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 20

4-4) SNL + Slice sampler with 10 chains
python3.6 SNL.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 10 --nsfTailBound 20

4-5) SNL + Metropolis-Hastings sampler with a single chain
python3.6 SNL.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 20

4-6) SNL + Metropolis-Hastings sampler with 100 chains
python3.6 SNL.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 100 --nsfTailBound 20

4-7) AALR + Slice sampler with a single chain
python3.6 AALR.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 20

4-8) AALR + Slice sampler with 10 chains
python3.6 AALR.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 10 --nsfTailBound 20

4-9) AALR + Metropolis-Hastings sampler with a single chain
python3.6 AALR.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 20

4-10) AALR + Metropolis-Hastings sampler with 10 chains
python3.6 AALR.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod no --device cuda:0 --numChains 100 --nsfTailBound 20

4-11) SNL + Implicit Surrogate Proposal
python3.6 SNL.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod rkl --device cuda:0 --nsfTailBound 20

4-12) AALR + Implicit Surrogate Proposal
python3.6 AALR.py --simulation mg1 --thetaDim 3 --xDim 5 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 100 --numTime 50 --numRound 10 --numModes 1 --posteriorInferenceMethod rkl --device cuda:0 --nsfTailBound 20

5) Competitive Lotka Volterra

5-1) SMC-ABC
python3.6 SBI.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numTime 1000 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --algorithm SMC --nsfTailBound 3

5-2) APT
python3.6 SBI.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numRound 10 --numTime 1000 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --algorithm SNPE --nsfTailBound 3

5-3) SNL + Slice sampler with a single chain
python3.6 SNL.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 3

5-4) SNL + Slice sampler with 10 chains
python3.6 SNL.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 10 --nsfTailBound 3

5-5) SNL + Metropolis-Hastings sampler with a single chain
python3.6 SNL.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 3

5-6) SNL + Metropolis-Hastings sampler with 100 chains
python3.6 SNL.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 100 --nsfTailBound 3

5-7) AALR + Slice sampler with a single chain
python3.6 AALR.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 3

5-8) AALR + Slice sampler with 10 chains
python3.6 AALR.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType sbiSliceSampler --burnInMCMC 100 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 10 --nsfTailBound 3

5-9) AALR + Metropolis-Hastings sampler with a single chain
python3.6 AALR.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 1 --nsfTailBound 3

5-10) AALR + Metropolis-Hastings sampler with 10 chains
python3.6 AALR.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod no --device cuda:0 --numChains 100 --nsfTailBound 3

5-11) SNL + Implicit Surrogate Proposal
python3.6 SNL.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod rkl --device cuda:0 --nsfTailBound 3

5-12) AALR + Implicit Surrogate Proposal
python3.6 AALR.py --simulation competitiveLotkaVolterra_ver2 --thetaDim 8 --xDim 10 --samplerType MHGaussianMultiChainsSampler --burnInMCMC 1000 --plotMIS False --plotPerformance True --logCount True --simulation_budget_per_round 1000 --numTime 1000 --numRound 10 --numModes 2 --posteriorInferenceMethod rkl --device cuda:0 --nsfTailBound 3
```







