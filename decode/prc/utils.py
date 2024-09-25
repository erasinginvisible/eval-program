import sys
import numpy as np
import torch
from ldpc import bp_decoder
from scipy.special import erf
import galois
from torchvision import transforms

GF = galois.GF(2)


def boolean_row_reduce(A, print_progress=False):
    n, k = A.shape
    A_rr = A.copy()
    perm = np.arange(n)
    for j in range(k):
        idxs = j + np.nonzero(A_rr[j:, j])[0]
        if idxs.size == 0:
            print("The given matrix is not invertible")
            return None
        A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]
        (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])
        A_rr[idxs[1:]] += A_rr[j]
        if print_progress:
            sys.stdout.write(f"\rDecoding progress: {j + 1} / {k}")
            sys.stdout.flush()
    if print_progress:
        print()
    return perm[:k]


def Detect(decoding_key, posteriors, false_positive_rate=None):
    (
        generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate_key,
        noise_rate,
        test_bits,
        g,
        max_bp_iter,
        t,
    ) = decoding_key
    if false_positive_rate is not None:
        fpr = false_positive_rate
    else:
        fpr = false_positive_rate_key

    posteriors = (
        (1 - 2 * noise_rate)
        * (1 - 2 * np.array(one_time_pad, dtype=float))
        * posteriors.numpy(force=True)
    )

    r = parity_check_matrix.shape[0]
    Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + Pi) / 2)
    log_minus = np.log((1 - Pi) / 2)
    log_prod = log_plus + log_minus

    const = 0.5 * np.sum(
        np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2)
    )
    threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()

    return log_plus.sum() >= threshold


def Decode(decoding_key, posteriors, print_progress=False):
    (
        generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate_key,
        noise_rate,
        test_bits,
        g,
        max_bp_iter,
        t,
    ) = decoding_key

    posteriors = (
        (1 - 2 * noise_rate)
        * (1 - 2 * np.array(one_time_pad, dtype=float))
        * posteriors.numpy(force=True)
    )
    channel_probs = (1 - np.abs(posteriors)) / 2
    x_recovered = (1 - np.sign(posteriors)) // 2

    if print_progress:
        print("Running belief propagation...")
    bpd = bp_decoder(
        parity_check_matrix,
        channel_probs=channel_probs,
        max_iter=max_bp_iter,
        bp_method="product_sum",
    )
    x_decoded = bpd.decode(x_recovered)

    bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
    confidences = 2 * np.abs(0.5 - bpd_probs)

    confidence_order = np.argsort(-confidences)
    ordered_generator_matrix = generator_matrix[confidence_order]
    ordered_x_decoded = x_decoded[confidence_order]

    top_invertible_rows = boolean_row_reduce(
        ordered_generator_matrix, print_progress=print_progress
    )
    if top_invertible_rows is None:
        return None

    if print_progress:
        print("Solving linear system...")
    recovered_string = np.linalg.solve(
        ordered_generator_matrix[top_invertible_rows],
        GF(ordered_x_decoded[top_invertible_rows]),
    )

    if not (recovered_string[: len(test_bits)] == test_bits).all():
        return None
    return np.array(recovered_string[len(test_bits) + g :])


def prc_gaussians(z, basis=None, variances=None):
    if variances is None:
        denominators = 4 * np.sqrt(3) * torch.ones_like(z)
    elif type(variances) is float:
        denominators = np.sqrt(2 * variances * (1 + variances))
    else:
        denominators = torch.sqrt(2 * variances * (1 + variances))

    if basis is None:
        return erf(z / denominators)
    else:
        return erf((z @ basis) / denominators)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0
