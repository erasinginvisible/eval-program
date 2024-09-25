import math
import random
import torch
import torch.nn as nn


class ImageShuffler:
    def __init__(self, splits, shuffle_indices, transform_indices=None):
        self.splits = splits  # Number of splits per dimension
        self.shuffle_indices = shuffle_indices  # Fixed order for shuffling
        self.transform_indices = transform_indices  # Fixed order for transformations
        assert len(shuffle_indices) == splits * splits, "Invalid shuffle indices length"
        if transform_indices is not None:
            assert (
                len(transform_indices) == splits * splits
            ), "Invalid transform indices length"

    def transform_square(self, square, index, reverse=False):
        if index == 0:
            return torch.rot90(
                square, k=1 if not reverse else 3, dims=[1, 2]
            )  # 90-degree rotation
        elif index == 1:
            return torch.rot90(
                square, k=2, dims=[1, 2]
            )  # 180-degree rotation (same for reverse)
        elif index == 2:
            return torch.rot90(
                square, k=3 if not reverse else 1, dims=[1, 2]
            )  # 270-degree rotation
        elif index == 3:
            return torch.flip(square, dims=(2,))  # horizontal flip (same for reverse)
        elif index == 4:
            return torch.flip(square, dims=(1,))  # vertical flip (same for reverse)
        else:
            raise ValueError(f"Invalid index: {index}")

    def random_exchange(self, arr, num_flips):
        # Check if the input for num_flips is valid
        if num_flips > len(arr) // 2:
            raise ValueError(
                "num_flips cannot be greater than half the length of the array"
            )

        # Create a set to keep track of indices that have been used
        used_indices = set()

        for _ in range(num_flips):
            # Randomly select two distinct indices
            while True:
                idx1, idx2 = random.sample(range(len(arr)), 2)
                # Ensure we haven't used these indices before
                if idx1 not in used_indices and idx2 not in used_indices:
                    break
            # Swap the elements at these indices
            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
            # Mark these indices as used
            used_indices.add(idx1)
            used_indices.add(idx2)

        return arr

    def shuffle(self, batch, update_shuffle_indices=False, mismatch_number=None):
        assert batch.size(2) == batch.size(3), "Images must be square"
        if update_shuffle_indices:
            if mismatch_number == None:
                self.shuffle_indices = [
                    random.sample(range(self.splits**2), self.splits**2)
                    for _ in range(batch.size(0))
                ]
            else:
                self.shuffle_indices = [
                    self.random_exchange(self.shuffle_indices[i], mismatch_number)
                    for i in range(batch.size(0))
                ]
            self.transform_indices = [
                [random.choice(range(5)) for _ in range(self.splits**2)]
                for _ in range(batch.size(0))
            ]
        elif isinstance(self.shuffle_indices[0], int):
            self.shuffle_indices = [self.shuffle_indices for _ in range(batch.size(0))]
            self.transform_indices = [
                self.transform_indices for _ in range(batch.size(0))
            ]

        batch_size = batch.size(0)
        side_length = batch.size(2) // self.splits
        multiplier = batch_size // len(self.shuffle_indices)

        recombined_images = []

        # print("shuffle:", self.shuffle_indices)
        for k in range(multiplier):
            for b in range(len(self.shuffle_indices)):
                image = batch[b + k * len(self.shuffle_indices)]

                squares = [
                    image[
                        :,
                        i * side_length : (i + 1) * side_length,
                        j * side_length : (j + 1) * side_length,
                    ]
                    for i in range(self.splits)
                    for j in range(self.splits)
                ]

                transformed_squares = [
                    self.transform_square(sq, self.transform_indices[b][i])
                    for i, sq in enumerate(squares)
                ]
                shuffled_squares = [
                    transformed_squares[i] for i in self.shuffle_indices[b]
                ]

                recombined_rows = [
                    torch.cat(shuffled_squares[i : i + self.splits], dim=2)
                    for i in range(0, self.splits * self.splits, self.splits)
                ]
                recombined_image = torch.cat(recombined_rows, dim=1)

                recombined_images.append(recombined_image.unsqueeze(0))

        recombined_batch = torch.cat(recombined_images, dim=0)
        return recombined_batch


def max_gcd_less_than_32(num):
    max_gcd_val = 1
    for i in range(2, 33, 2):
        current_gcd = math.gcd(num, i)
        max_gcd_val = max(max_gcd_val, current_gcd)
    return max_gcd_val


def replace_batchnorm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child: torch.nn.BatchNorm2d = child
            setattr(
                module,
                name,
                nn.GroupNorm(
                    max_gcd_less_than_32(child.num_features), child.num_features
                ),
            )
        else:
            replace_batchnorm(child)
