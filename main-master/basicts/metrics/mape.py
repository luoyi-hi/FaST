import numpy as np
import torch


def masked_mape_bts(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Percentage Error (MAPE) between predicted and target values,
    ignoring entries that are either zero or match the specified null value in the target tensor.

    This function is particularly useful for time series or regression tasks where the target values may
    contain zeros or missing values, which could otherwise distort the error calculation. The function
    applies a mask to ensure these entries do not affect the resulting MAPE.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute percentage error.

    Details:
        - The function creates two masks:
          1. `zero_mask`: This mask excludes entries in the `target` tensor that are close to zero,
             since division by zero or near-zero values would result in extremely large or undefined errors.
          2. `null_mask`: This mask excludes entries in the `target` tensor that match the specified `null_val`.
             If `null_val` is `np.nan`, the mask will exclude `NaN` values using `torch.isnan`.

        - The final mask is the intersection of `zero_mask` and `null_mask`, ensuring that only valid, non-zero,
          and non-null values contribute to the MAPE calculation.
    """

    # mask to exclude zero values in the target
    zero_mask = ~torch.isclose(target, torch.tensor(0.0).to(target.device), atol=5e-5)

    # mask to exclude null values in the target
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    # combine zero and null masks
    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs((prediction - target) / target)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)


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
