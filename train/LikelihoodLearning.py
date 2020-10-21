import torch
import train.networks as networks
import train.train as train

def LikelihoodLearning(args, round, thetas, simulated_output, training_theta, training_x,
                       validation_theta, validation_x, netLikelihood, optLikelihood):
    # Put Simulation Input and Output into Repository
    permutation = torch.randperm(args.simulation_budget_per_round)
    if round == 0:
        permutation = torch.randperm(thetas.shape[0])
        training_theta = thetas[
            permutation[int(args.validationRatio * thetas.shape[0]):]]
        training_x = simulated_output[
            permutation[int(args.validationRatio * thetas.shape[0]):]]
        validation_theta = thetas[
            permutation[:int(args.validationRatio * thetas.shape[0])]]
        validation_x = simulated_output[
            permutation[:int(args.validationRatio * thetas.shape[0])]]

    elif round > 0:
        training_theta = torch.cat((training_theta, thetas[
            permutation[int(args.validationRatio * args.simulation_budget_per_round):]]))
        training_x = torch.cat((training_x, simulated_output[
            permutation[int(args.validationRatio * args.simulation_budget_per_round):]]))
        validation_theta = torch.cat((validation_theta, thetas[
            permutation[:int(args.validationRatio * args.simulation_budget_per_round)]]))
        validation_x = torch.cat((validation_x, simulated_output[
            permutation[:int(args.validationRatio * args.simulation_budget_per_round)]]))

    # Make Train Loader for Likelihood Learning
    train_loader = torch.utils.data.DataLoader(
        torch.cat((training_theta, training_x), 1), batch_size=args.batch_size, shuffle=True)

    # Re-initialize Likelihood Network
    if args.likelihoodParameterInitialize:
        netLikelihood, optLikelihood = networks.likelihoodNetwork(args)

    # Likelihood learning
    converged = False
    numEpoch = 0
    best_validation_likelihood = 1e10
    validation_tolerance = 0
    netLikelihood.train()
    while not converged:
        converged, validation_tolerance, best_validation_likelihood = train.trainLikelihood(args, train_loader,
                                                                                            validation_theta,
                                                                                            validation_x,
                                                                                            netLikelihood,
                                                                                            optLikelihood,
                                                                                            best_validation_likelihood,
                                                                                            validation_tolerance,
                                                                                            numEpoch)
        numEpoch += 1
    netLikelihood.eval()

    return training_theta, training_x, validation_theta, validation_x, netLikelihood