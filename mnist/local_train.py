import numpy as np
from numpy.random import randn as randn
import random
import torch
from PIL import Image
from tqdm import tqdm

import numpy as np

def find_interval(time_slices, t):
    for ti, tf in time_slices:
        if ti < t <= tf:
            return ti
    return None  

def rel_to_abs_time(time_slices, t):
    durations = [tf-ti for ti, tf in time_slices]
    total_duration = np.sum(durations)
    tt = t*total_duration

    i = 0
    while(tt>0):
        tt -= durations[i]
        i += 1
    
    if i == 0:
        return None
    ti = time_slices[i-1][0]
    tt = time_slices[i-1][1] + tt
    return [ti, tt]

def get_complementary_slices(time_slices):
    complement = []
    current = 0.0

    for ti, tf in time_slices:
        if ti > current:
            complement.append((current, ti))
        current = max(current, tf)

    if current < 1.0:
        complement.append((current, 1.0))

    return complement

def train(epoch, model_global, optimizer_global, model_local, optimizer_local, local_time_slices, loss, dataloader, DEVICE, EPOCHS, CLASSES, train_with_cond=False):
    model_local.train()
    model_global.train()
    ema_loss = 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01, disable=True) as pbar:
        for i, (img, cond) in enumerate(dataloader):
            global_time_slices = get_complementary_slices(local_time_slices)
            # img ~ Image, z ~ N

            img = img.to(DEVICE)
            cond = cond.to(DEVICE).long()
            z = torch.randn_like(img).to(DEVICE)
            
            #### global ####
                        
            # t ~ global slices :
            #   (training)
            #   img_t = img_0 + (z - img_0) * t
            #   model_gloabl(img_t, t) = img_0 - z
            #   (generation)
            #   img_t-dt = img_t + model_global(img_t, t) * dt
            
            t = torch.sigmoid(torch.randn(img.shape[0]))
            t_global = torch.tensor([rel_to_abs_time(global_time_slices, tt) for tt in t]).to(DEVICE)[:,1].view(-1, 1, 1, 1)
            img_t = img * (1 - t_global) + z * t_global
            target = img - z

            optimizer_global.zero_grad()
            if train_with_cond:
                y = model_global(img_t.view(img.shape[0], -1), t_global.view(-1, 1), cond).view(img.shape)
            else:
                cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype) 
                y = model_global(img_t.view(img.shape[0], -1), t_global.view(-1, 1), cond_zero).view(img.shape)
            eps_pred = img_t - (1 - t_global) * y
            img0_pred = img_t + t_global * y

            l_global = loss(y, target) + loss(eps_pred, z) + loss(img0_pred, img)
            l_global.backward()
            optimizer_global.step()

            #### local ####
            # t ~ local slices :
            #   (training)
            #   find ti of t
            #   z_accumulated = 0
            #   for ly in [-L, L]
            #       for lx in [-L, L]
            #           z_l = push_A(pull_A(z, lx, ly))
            #           z_accumulated += zl
            #           img_t_l = img_0 + ti * (z-img0) + (t-ti) * z_accumulated
            #           model_local(pull_B(img_t_l, lx, ly), t, lx, ly) = pull_A(img_0 - z)
            #   (generation)
            #   d_img = 0
            #   for ly in [L, -L]
            #       for lx in [L, -L]
            #           d_img += push_A(model_local(img_t + (t-ti) * d_img, t, lx, ly) )
            #   img_t-dt = img_t + d_img * dt
            # logit-norm time schedule

            t = torch.sigmoid(torch.randn(img.shape[0]))
            tt_local = torch.tensor([rel_to_abs_time(local_time_slices, tt) for tt in t]).to(DEVICE)
            ti_local = tt_local[:,0].view(-1, 1, 1, 1)
            t_local = tt_local[:,1].view(-1, 1, 1, 1)
            
            #l_local = 0
            z_accumulated = torch.zeros_like(img).to(DEVICE)
            # for ly in range(-model_local.ls.L, model_local.ls.LL+1):
            #     for lx in range(-model_local.ls.L, model_local.ls.LL+1):
            ly = random.randint(-model_local.ls.L, model_local.ls.LL)
            lx = random.randint(-model_local.ls.L, model_local.ls.LL)


            img_A = model_local.ls.pull_layer_A(img, lx, ly)
            z_A = model_local.ls.pull_layer_A(z, lx, ly)
            z_l = model_local.ls.push_layer_A(z_A, lx, ly).to(DEVICE)
            z_accumulated += z_l
            img_t_l = img + ti_local * (z - img) + (t_local - ti_local) * (z_accumulated - img)

            patches = model_local.ls.pull_layer_B(img_t_l, lx, ly)
            lx_gpu = torch.tensor(lx).to(DEVICE).long()
            ly_gpu = torch.tensor(ly).to(DEVICE).long()
            if train_with_cond:
                y = model_local(patches, ti_local, t_local, lx_gpu, ly_gpu, cond)
            else:
                cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype)
                y = model_local(patches, ti_local, t_local, lx_gpu, ly_gpu, cond_zero)

            target = img_A - z_A
            img_t_A = model_local.ls.pull_layer_A(img_t_l, lx, ly)
            eps_pred =  img_t_A - (1 - t_local.view(-1,1,1)) * y
            img0_pred = img_t_A + t_local.view(-1,1,1) * y

            l_local = loss(y, target) + loss(eps_pred, z_A) + loss(img0_pred, img_A)

            l_local.backward()
            optimizer_local.step()

            # all_grads = []
            # for param in model.parameters():
            #     if param.grad is not None:
            #         all_grads.append(param.grad.view(-1))  # Flatten and collect

            # # Compute average and max gradient
            # if all_grads:
            #     all_grads = torch.cat(all_grads)  # Concatenate all gradients into one tensor
            #     avg_grad = all_grads.mean().item()  # Compute mean
            #     max_grad = all_grads.abs().max().item()  # Compute max absolute gradient
            #     print(f"Epoch {epoch+1}: Avg Gradient = {avg_grad:.6f}, Max Gradient = {max_grad:.6f}")

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * (l_global.item() + l_local.item())
            
            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})
    
    torch.save(model_global.state_dict(), "mnist-global.pth")
    torch.save(model_local.state_dict(), "mnist-local.pth")


def test(epoch, model_global, model_local, local_time_slices, DEVICE, EPOCHS, CLASSES, gen_with_cond=False, T=32):
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    model_global.eval()
    model_local.eval()
    IMAGE_COUNT = 16 * 16
    with torch.no_grad():
        img_t = torch.randn(IMAGE_COUNT, model_local.ls.c, model_local.ls.h, model_local.ls.w).to(DEVICE)
        img_shape = img_t.shape
        cond = torch.arange(IMAGE_COUNT).long().to(DEVICE) % CLASSES

        t = 1
        while (t>0):
            ti = find_interval(local_time_slices, t)
            if ti is None:
                print(t, 'global')
                # global
                dt = min(t-0, 1/T)
                tt = torch.tensor(t).view(1,1).repeat(IMAGE_COUNT,1).to(DEVICE)
                if gen_with_cond:
                    y = model_global(img_t.view(IMAGE_COUNT, -1), tt, cond).view(img_shape)
                else:
                    cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype)
                    y = model_global(img_t.view(IMAGE_COUNT, -1), tt, cond_zero).view(img_shape)
                img_t = img_t + dt * y
                # img_t = img_t + dt * model_global(img_t.view(IMAGE_COUNT, -1), tt, cond).view(img_shape)
                t -= dt
            else:
                # local
                print(t, 'local')
                dt = min(t-ti, 1/T)
                d_img = torch.zeros(img_shape).to(DEVICE)

                for ly in range(model_local.ls.LL, -model_local.ls.L-1, -1):
                    for lx in range(model_local.ls.LL, -model_local.ls.L-1, -1):
                        lx_gpu = torch.tensor(lx).to(DEVICE).long()
                        ly_gpu = torch.tensor(ly).to(DEVICE).long()
                        tti = torch.tensor(ti).view(1,1).repeat(IMAGE_COUNT,1).to(DEVICE)
                        tt = torch.tensor(t).view(1,1).repeat(IMAGE_COUNT,1).to(DEVICE)
                        patches = model_local.ls.pull_layer_B(img_t + (t-ti) * d_img, lx, ly)
                        if gen_with_cond:
                            y = model_local(patches, tti, tt, lx_gpu, ly_gpu, cond)
                        else:
                            cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype)
                            y = model_local(patches, tti, tt, lx_gpu, ly_gpu, cond_zero)
                        # y = model_local(patches, tti, tt, lx_gpu, ly_gpu, cond)
                        d_img += model_local.ls.push_layer_A(y, lx, ly)
                
                img_t += dt * d_img
                t -= dt

    #print(torch.min(pred), torch.max(pred))
    img_t = img_t.reshape(16, 16, 28, 28).permute(0, 2, 1, 3) * 0.5 + 0.5
    img_t = img_t.reshape(16 * 28, 16 * 28).cpu().numpy()
    img_t = (img_t * 255).clip(0, 255).astype(np.uint8)
    img_t = Image.fromarray(img_t)
    img_t.save(f"./mnist-results_local_alphablend/gen-{epoch}.png")
    torch.set_rng_state(rng_state)



def train_local(epoch, model_local, optimizer_local, local_time_slices, loss, dataloader, DEVICE, EPOCHS, CLASSES, train_with_cond=False):
    model_local.train()
    ema_loss = 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01, miniters=1000) as pbar:
        for i, (img, cond) in enumerate(dataloader):
            img = img.to(DEVICE)
            cond = cond.to(DEVICE).long()
            z = torch.randn_like(img).to(DEVICE)

            t = torch.sigmoid(torch.randn(img.shape[0]))
            tt_local = torch.tensor([rel_to_abs_time(local_time_slices, tt) for tt in t]).to(DEVICE)
            ti_local = tt_local[:,0].view(-1, 1, 1, 1)
            t_local = tt_local[:,1].view(-1, 1, 1, 1)
            
            #l_local = 0
            z_accumulated = torch.zeros_like(img).to(DEVICE)
            # for ly in range(-model_local.ls.L, model_local.ls.LL+1):
            #     for lx in range(-model_local.ls.L, model_local.ls.LL+1):
            ly = random.randint(-model_local.ls.L, model_local.ls.LL)
            lx = random.randint(-model_local.ls.L, model_local.ls.LL)


            img_A = model_local.ls.pull_layer_A(img, lx, ly)
            z_A = model_local.ls.pull_layer_A(z, lx, ly)
            z_l = model_local.ls.push_layer_A(z_A, lx, ly).to(DEVICE)
            z_accumulated += z_l
            img_t_l = img + ti_local * (z - img) + (t_local - ti_local) * (z_accumulated - img)

            patches = model_local.ls.pull_layer_B(img_t_l, lx, ly)
            lx_gpu = torch.tensor(lx).to(DEVICE).long()
            ly_gpu = torch.tensor(ly).to(DEVICE).long()
            if train_with_cond:
                y = model_local(patches, ti_local, t_local, lx_gpu, ly_gpu, cond)
            else:
                cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype)
                y = model_local(patches, ti_local, t_local, lx_gpu, ly_gpu, cond_zero)
            # y = model_local(patches, ti_local, t_local, lx_gpu, ly_gpu, cond if train_with_cond else None)


            target = img_A - z_A
            img_t_A = model_local.ls.pull_layer_A(img_t_l, lx, ly)
            eps_pred =  img_t_A - (1 - t_local.view(-1,1,1)) * y
            img0_pred = img_t_A + t_local.view(-1,1,1) * y

            l_local = loss(y, target) + loss(eps_pred, z_A) + loss(img0_pred, img_A)

            l_local.backward()
            optimizer_local.step()

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * (l_local.item())

            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})


def train_partial(epoch, model_local, optimizer_local, loss, dataloader, tf, DEVICE, EPOCHS, CLASSES, train_with_cond=False):
    """
    Train GenLocal to denoise images with noise added up to time tf.
    """
    model_local.train()
    ema_loss = 0

    with tqdm(dataloader, desc=f"Partial Epoch {epoch}", smoothing=0.01,miniters=100) as pbar:
        for i, (img, cond) in enumerate(dataloader):
            img = img.to(DEVICE)
            cond = cond.to(DEVICE).long()
            z = torch.randn_like(img).to(DEVICE)
            t = torch.full((img.shape[0], 1, 1, 1), tf, device=DEVICE)
            img_noisy = img * (1 - t) + z * t

            # Random patch position
            ly = random.randint(-model_local.ls.L, model_local.ls.LL)
            lx = random.randint(-model_local.ls.L, model_local.ls.LL)

            img_A = model_local.ls.pull_layer_A(img, lx, ly)
            img_noisy_B = model_local.ls.pull_layer_B(img_noisy, lx, ly)

            lx_gpu = torch.tensor(lx).to(DEVICE).long()
            ly_gpu = torch.tensor(ly).to(DEVICE).long()
            t_tensor = torch.full((img.shape[0], 1, 1), tf, device=DEVICE)
            ti_tensor = torch.zeros_like(t_tensor)

            optimizer_local.zero_grad()
            if train_with_cond:
                y = model_local(img_noisy_B, ti_tensor, t_tensor, lx_gpu, ly_gpu, cond)
            else:
                cond_zero = torch.zeros(cond.shape, device=DEVICE, dtype=cond.dtype)
                y = model_local(img_noisy_B, ti_tensor, t_tensor, lx_gpu, ly_gpu, cond_zero)
            # y = model_local(img_noisy_B, ti_tensor, t_tensor, lx_gpu, ly_gpu, cond if train_with_cond else None)
            l = loss(y, img_A)
            l.backward()
            optimizer_local.step()

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * l.item()
            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})


def test_partial(model_local, dataset, tf, loss, DEVICE, EPOCHS, CLASSES, IMAGE_COUNT=16*16, save_path="./mnist-results_local_alphablend/partial_denoise.png", with_cond=True, T=32):
    """
    Test GenLocal by denoising images with noise added up to time tf, using a time-stepping process like the main test function.
    """
    # fix seed for reproducibility
    torch.manual_seed(0)
    # set rnd_state to ensure consistent results
    rnd_state = torch.get_rng_state()

    model_local.eval()
    with torch.no_grad():
        # Sample IMAGE_COUNT images from the dataset
        indices = torch.randperm(len(dataset))[:IMAGE_COUNT]
        imgs = []
        conds = []
        for idx in indices:
            img, cond = dataset[idx]
            imgs.append(img)
            conds.append(cond)
        imgs = torch.stack(imgs).to(DEVICE)
        conds = torch.tensor(conds).long().to(DEVICE)

        z = torch.randn_like(imgs).to(DEVICE)
        t = tf
        img_t = imgs * (1 - t) + z * t
        img_shape = imgs.shape

        # Denoising with time steps from tf down to 0
        while t > 0:
            dt = min(t, 1 / T)
            d_img = torch.zeros_like(img_t)
            for ly in range(model_local.ls.LL, -model_local.ls.L-1, -1):
                for lx in range(model_local.ls.LL, -model_local.ls.L-1, -1):
                    lx_gpu = torch.tensor(lx).to(DEVICE).long()
                    ly_gpu = torch.tensor(ly).to(DEVICE).long()
                    t_tensor = torch.full((IMAGE_COUNT, 1, 1), t, device=DEVICE)
                    ti_tensor = torch.zeros_like(t_tensor)
                    patches = model_local.ls.pull_layer_B(img_t + (t - 0) * d_img, lx, ly)
                    if with_cond:
                        y = model_local(patches, ti_tensor, t_tensor, lx_gpu, ly_gpu, conds)
                    else:
                        cond_zero = torch.zeros(conds.shape, device=DEVICE, dtype=conds.dtype)
                        y = model_local(patches, ti_tensor, t_tensor, lx_gpu, ly_gpu, cond_zero)
                    # y = model_local(patches, ti_tensor, t_tensor, lx_gpu, ly_gpu, conds if with_cond else None)
                    d_img += model_local.ls.push_layer_A(y, lx, ly)
            img_t = img_t + dt * d_img
            t -= dt

        # Save the denoised images as a grid
        img_out = img_t.reshape(16, 16, 28, 28).permute(0, 2, 1, 3) * 0.5 + 0.5
        img_out = img_out.reshape(16 * 28, 16 * 28).cpu().numpy()
        img_out = (img_out * 255).clip(0, 255).astype(np.uint8)
        img_out = Image.fromarray(img_out)
        img_out.save(save_path)

        # Compute average loss
        avg_loss = loss(img_t, imgs).item() / IMAGE_COUNT

        # Restore the random state
        torch.set_rng_state(rnd_state)
        return avg_loss
    
# just compute the loss, don't denoise the images
def test_partial_no_denoise(model_local, dataset, tf, loss, DEVICE, EPOCHS, CLASSES, IMAGE_COUNT=16*16, save_path="./mnist-results_local_alphablend/partial_denoise.png", with_cond=True, T=32):
    """
    Test GenLocal by denoising images with noise added up to time tf, using a time-stepping process like the main test function.
    """
    # fix seed for reproducibility
    torch.manual_seed(0)
    # set rnd_state to ensure consistent results
    rnd_state = torch.get_rng_state()

    model_local.eval()
    with torch.no_grad():
        # Sample IMAGE_COUNT images from the dataset
        indices = torch.randperm(len(dataset))[:IMAGE_COUNT]
        imgs = []
        conds = []
        for idx in indices:
            img, cond = dataset[idx]
            imgs.append(img)
            conds.append(cond)
        imgs = torch.stack(imgs).to(DEVICE)
        conds = torch.tensor(conds).long().to(DEVICE)

        z = torch.randn_like(imgs).to(DEVICE)
        t = tf
        img_t = imgs * (1 - t) + z * t
        img_shape = imgs.shape

        # Compute average loss
        avg_loss = loss(img_t, imgs).item() / IMAGE_COUNT

        # Restore the random state
        torch.set_rng_state(rnd_state)
        return avg_loss