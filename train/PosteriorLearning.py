import torch
import train.networks as networks
import train.train as train

def PosteriorLearning(args, simulation, teacher_theta, netPosterior=None):
    if teacher_theta != None:
        # Make Train Loader for Posterior Learning
        permutation = torch.randperm(args.num_training)
        training_teacher_theta = teacher_theta[permutation[int(0.5 * args.num_training):]]
        validation_teacher_theta = teacher_theta[permutation[:int(0.5 * args.num_training)]]
        print(training_teacher_theta.shape, validation_teacher_theta.shape)
        train_loader = torch.utils.data.DataLoader(training_teacher_theta, batch_size=args.batch_size, shuffle=True)

        # Posterior learning
        if (args.posteriorInferenceMethod != 'no' and args.posteriorParameterInitialize) or (netPosterior == None):
            netPosterior, optPosterior = networks.posteriorNetwork(args, simulation, teacher_theta)
        converged = False
        numEpoch = 0
        best_validation_loss = 1e10
        validation_tolerance = 0
        netPosterior.train()
        while not converged:
            converged, validation_tolerance, best_validation_loss = train.trainPosterior(args, train_loader,
                                                                                         validation_teacher_theta,
                                                                                         netPosterior,
                                                                                         optPosterior,
                                                                                         best_validation_loss,
                                                                                         validation_tolerance, numEpoch)
            numEpoch += 1
            #if numEpoch == 40:
            #    converged = True
        netPosterior.eval()

        return netPosterior
    else:
        return None