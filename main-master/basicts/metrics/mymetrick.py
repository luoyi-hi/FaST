import numpy as np
import torch


def masked_ae(
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
    # mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.sum(loss), torch.sum(mask)



def masked_se(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Squared Error (MSE) between predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is useful for scenarios where the dataset contains missing or irrelevant values
    (denoted by `null_val`) that should not contribute to the loss calculation. The function applies
    a mask to these values, ensuring they do not affect the error metric.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean squared error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    # mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.sum(loss), torch.sum(mask)  # Return the mean of the masked loss


def masked_ape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
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

    # mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs((prediction - target) / target)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.sum(loss), torch.sum(mask)


