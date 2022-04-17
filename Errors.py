import numpy as np

def compute_errors(gt, pred):
    newGt = np.interp(gt, (0, 1), (0, 255))
    newPred = np.interp(pred, (0, 1), (0, 255))
    
    newGt = newGt.astype(np.uint16)
    newPred = newPred.astype(np.uint16)
    
    newGt = newGt[newPred > 0]
    newPred = newPred[newPred > 0]
    
    newPred = newPred[newGt > 0]
    newGt = newGt[newGt > 0]
    
    thresh = np.maximum(np.true_divide(newGt, newPred), np.true_divide(newPred, newGt))
    delta1 = (thresh < 1.25   ).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    mse = ((newGt - newPred) ** 2).mean()
    rmse = np.sqrt(mse)
    
    abs_diff = np.abs(newGt - newPred)
    abs_rel = np.mean(abs_diff / newGt)
    mae = abs_diff.mean()

    rmse_log = (np.log(newGt) - np.log(newPred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    sq_rel = np.mean(((newGt - newPred)**2) / newGt)
    
    inv_pred = 1 / newPred
    inv_gt = 1 / newGt
    abs_inv_diff = np.abs(inv_pred - inv_gt)
    
    irmse = np.sqrt((abs_inv_diff ** 2).mean())
    imae = np.mean(abs_inv_diff)

    return mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3