import numpy as np
import torch

def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    device = prediction.device
    total_loss = 0.0
    total_weight = 0.0

    num_samples = prediction.size(0)

    batch_size_limit = 64

    if num_samples <= batch_size_limit:
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            null_tensor = torch.full_like(target, null_val, device=device)
            mask = ~torch.isclose(target, null_tensor, atol=eps, rtol=0.0)

        mask = mask.float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        mask = torch.nan_to_num(mask)

        loss = torch.abs(prediction - target)
        loss = loss * mask
        loss = torch.nan_to_num(loss)

        total_loss = loss.sum()
        total_weight = mask.sum()
    else:
        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            if np.isnan(null_val):
                mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_tensor = torch.full_like(target_batch, null_val, device=device)
                mask = ~torch.isclose(target_batch, null_tensor, atol=eps, rtol=0.0)

            mask = mask.float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)

            loss = torch.abs(pred_batch - target_batch)
            loss = loss * mask
            loss = torch.nan_to_num(loss)

            total_loss += loss.sum()
            total_weight += mask.sum()

            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_mape(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:

    device = prediction.device
    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    num_samples = prediction.size(0)
    batch_size_limit = 64

    if num_samples <= batch_size_limit:
        zero_mask = ~torch.isclose(target, torch.tensor(0.0, device=device), atol=5e-5)

        if np.isnan(null_val):
            null_mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            null_mask = ~torch.isclose(
                target, torch.tensor(null_val, device=device), atol=eps, rtol=0.0
            )

        mask = (zero_mask & null_mask).float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        mask = torch.nan_to_num(mask)

        loss = torch.abs((prediction - target) / target)
        loss = loss * mask
        loss = torch.nan_to_num(loss)

        total_loss = loss.sum()
        total_weight = mask.sum()
    else:
        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            zero_mask = ~torch.isclose(
                target_batch, torch.tensor(0.0, device=device), atol=5e-5
            )

            if np.isnan(null_val):
                null_mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_mask = ~torch.isclose(
                    target_batch,
                    torch.tensor(null_val, device=device),
                    atol=eps,
                    rtol=0.0,
                )

            mask = (zero_mask & null_mask).float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)

            loss = torch.abs((pred_batch - target_batch) / target_batch)
            loss = loss * mask
            loss = torch.nan_to_num(loss)

            total_loss += loss.sum()
            total_weight += mask.sum()

            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    device = prediction.device
    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    num_samples = prediction.size(0)

    batch_size_limit = 64

    if num_samples <= batch_size_limit:
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            null_tensor = torch.full_like(target, null_val, device=device)
            mask = ~torch.isclose(target, null_tensor, atol=eps, rtol=0.0)

        mask = mask.float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        mask = torch.nan_to_num(mask)

        loss = (prediction - target) ** 2
        loss = loss * mask
        loss = torch.nan_to_num(loss)

        total_loss += loss.sum()
        total_weight += mask.sum()
    else:

        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            if np.isnan(null_val):
                mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_tensor = torch.full_like(target_batch, null_val, device=device)
                mask = ~torch.isclose(target_batch, null_tensor, atol=eps, rtol=0.0)

            mask = mask.float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)

            loss = (pred_batch - target_batch) ** 2
            loss = loss * mask
            loss = torch.nan_to_num(loss)
            total_loss += loss.sum()
            total_weight += mask.sum()

            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))
