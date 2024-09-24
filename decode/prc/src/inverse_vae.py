from diffusers import AutoencoderKL
import torch, copy

class InversableVAE(AutoencoderKL):
    def __init__(self,
        down_block_types,
        up_block_types,
        block_out_channels,
        layers_per_block,
    ):
        super(InversableVAE, self).__init__(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents


    def decoder_inv(self, x):
        """
        decoder_inv calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512)
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z)
        """        
        input = x.clone().float()
        z = self.get_image_latents(x).clone().float()
        z.requires_grad_(True)

        loss_function = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam([z], lr=0.01) 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        for n in range(20):  # Only 20 steps
            scaled_latents = 1 / 0.18215 * z
            new_vae = copy.deepcopy(self).float()
            image = [
                new_vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(z))
            ]
            x_pred = torch.cat(image, dim=0)

            loss = loss_function(x_pred, input)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(z, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            torch.cuda.empty_cache() 

        return z
   
        