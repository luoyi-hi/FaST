import numpy as np
import torch


def masked_mae_bts(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val).expand_as(target).to(target.device),
            atol=eps,
            rtol=0.0,
        )

    mask = mask.float()
    mask /= torch.mean(
        mask
    )  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


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
