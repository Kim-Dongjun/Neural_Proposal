import torch
import numpy as np


def trainLikelihood(args, train_loader, validation_theta, validation_x, netLikelihood, optLikelihood, best_validation_likelihood, validation_tolerance, numEpoch):

    for batch_idx, data in enumerate(train_loader):
        training_theta = data[:,:args.thetaDim]
        training_x = data[:,args.thetaDim:]
        #print("training_x : ", training_x)
        if args.likelihoodLearningDecay:
            #optLikelihood.param_groups[0]["lr"] = 0.02 * pow(0.9, numEpoch)
            optLikelihood.param_groups[0]["lr"] = 0.002# * pow(0.95, numEpoch)
        optLikelihood.zero_grad()
        if args.likelihoodFlowType == 'nflow_maf' or args.likelihoodFlowType == 'nsf':
            lossLikelihood = - torch.clamp(netLikelihood.log_prob(
                context=training_theta, inputs=training_x), -1e8).mean()
        elif args.likelihoodFlowType == 'naf':
            lossLikelihood = - torch.clamp(
                netLikelihood.condition(training_theta).log_prob(training_x), -1e8).mean()
        elif args.likelihoodFlowType != 'naf':
            lossLikelihood = - torch.clamp(netLikelihood.log_probs(
                    context=training_theta, inputs=training_x), -1e8).mean()

        lossLikelihood.backward(retain_graph=True)
        optLikelihood.step()

    if args.likelihoodFlowType == 'nflow_maf' or args.likelihoodFlowType == 'nsf':
        current_epoch_validation_likelihood = - torch.clamp(netLikelihood.log_prob(
            context=validation_theta, inputs=validation_x), -1e8).mean().item()
    elif args.likelihoodFlowType == 'naf':
        current_epoch_validation_likelihood = - torch.clamp(
            netLikelihood.condition(validation_theta).log_prob(validation_x), -1e8).mean().item()
    elif args.likelihoodFlowType != 'naf':
        current_epoch_validation_likelihood = - torch.clamp(netLikelihood.log_probs(
            context=validation_theta, inputs=validation_x), -1e8).mean().item()
    print(str(numEpoch) + "-th Epoch Likelihood (training, validation, best validation) losses : ", lossLikelihood.item(), current_epoch_validation_likelihood, best_validation_likelihood, optLikelihood.param_groups[0]["lr"])
    if best_validation_likelihood > current_epoch_validation_likelihood:
        best_validation_likelihood = current_epoch_validation_likelihood
        validation_tolerance = 0
    else:
        validation_tolerance += 1
    if validation_tolerance == args.maxValidationTolerance:
        return True, validation_tolerance, best_validation_likelihood
    else:
        return False, validation_tolerance, best_validation_likelihood

def trainDiscriminator(args, train_loader, validation_theta, validation_x, netDiscriminator, optDiscriminator,
                       best_validation_likelihood, validation_tolerance, numEpoch):
    for batch_idx, data in enumerate(train_loader):
        training_theta = data[:, :args.thetaDim]
        training_x = data[:, args.thetaDim:]
        optDiscriminator.zero_grad()
        lossDiscriminator = BCELoss(netDiscriminator, training_theta, training_x)
        #print("loss : ", lossDiscriminator)
        lossDiscriminator.backward(retain_graph=True)
        optDiscriminator.step()
        #print("loss : ", lossDiscriminator.item())

    current_epoch_validation_likelihood = BCELoss(netDiscriminator, validation_theta, validation_x).item()
    print(str(numEpoch) + "-th Epoch Likelihood (training, validation, best validation) losses : ",
          lossDiscriminator.item(), current_epoch_validation_likelihood, best_validation_likelihood,
          optDiscriminator.param_groups[0]["lr"])
    if best_validation_likelihood > current_epoch_validation_likelihood:
        best_validation_likelihood = current_epoch_validation_likelihood
        validation_tolerance = 0
    else:
        validation_tolerance += 1
    if validation_tolerance == args.maxValidationTolerance:
        return True, validation_tolerance, best_validation_likelihood
    else:
        return False, validation_tolerance, best_validation_likelihood

def BCELoss(netDiscriminator, theta, x):
    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]

    logits = _classifier_logits(netDiscriminator, theta, x, 2)
    likelihood = torch.sigmoid(logits).squeeze()
    #print(likelihood)

    if logits.get_device() < 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(logits.get_device())

    # Alternating pairs where there is one sampled from the joint and one
    # sampled from the marginals. The first element is sampled from the
    # joint p(theta, x) and is labelled 1. The second element is sampled
    # from the marginals p(theta)p(x) and is labelled 0. And so on.
    labels = torch.ones(2 * batch_size).to(device)  # two atoms
    labels[1::2] = 0.0
    #print("!! : ", likelihood, labels)
    # Binary cross entropy to learn the likelihood (AALR-specific)
    return torch.nn.BCELoss()(likelihood, labels)

def _classifier_logits(netDiscriminator, theta, x, num_atoms=2):
    """Return logits obtained through classifier forward pass.

    The logits are obtained from atomic sets of (theta,x) pairs.
    """
    batch_size = theta.shape[0]
    repeated_x = repeat_rows(x, num_atoms)

    # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
    probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)

    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

    contrasting_theta = theta[choices]

    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )

    theta_and_x = torch.cat((atomic_theta, repeated_x), dim=1)
    #for k in range(theta_and_x.shape[0]):
    #    print("!! : ", k, theta_and_x[k])
    return netDiscriminator(theta_and_x)

def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)

def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)

def trainPosterior(args, train_loader, validation_teacher_data, netPosterior, optPosterior,
                   best_validation_loss, validation_tolerance, numEpoch):
    if validation_teacher_data != None:
        for batch_idx, data in enumerate(train_loader):
            optPosterior.zero_grad()
            if args.posteriorLearningDecay:
                if optPosterior.param_groups[0]["lr"] > 1e-5:
                    optPosterior.param_groups[0]["lr"] = 0.01 * pow(0.93, numEpoch)
            if args.posteriorFlowType == 'nflow_maf' or args.posteriorFlowType == 'nsf':
                lossPosterior = - torch.clamp(netPosterior.log_prob(
                    data), -1e8).mean()
            else:
                lossPosterior = - torch.clamp(
                                    netPosterior.log_probs(data), -1e8).mean()
            lossPosterior.backward(retain_graph=True)
            optPosterior.step()
        if args.posteriorFlowType == 'nflow_maf' or args.posteriorFlowType == 'nsf':
            current_epoch_validation_loss = - torch.clamp(netPosterior.log_prob(validation_teacher_data), -1e8).mean()
        else:
            current_epoch_validation_loss = - torch.clamp(netPosterior.log_probs(validation_teacher_data), -1e8).mean()

        if best_validation_loss > current_epoch_validation_loss:
            best_validation_loss = current_epoch_validation_loss
            validation_tolerance = 0
        else:
            validation_tolerance += 1
        print(str(numEpoch) + "-th Epoch Posterior (training, validation, best validation) losses : ",
              lossPosterior.item(),
              current_epoch_validation_loss.item(), best_validation_loss.item(), optPosterior.param_groups[0]["lr"])
        if validation_tolerance == 20:
            return True, validation_tolerance, best_validation_loss
        else:
            return False, validation_tolerance, best_validation_loss

    else:
        return