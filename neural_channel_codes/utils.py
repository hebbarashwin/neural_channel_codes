import torch
import numpy as np

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def errors_ber(y_true, y_pred, mask=None):
    if mask == None:
        mask=torch.ones(y_true.size(),device=y_true.device)
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    mask = mask.view(mask.shape[0], -1, 1)
    myOtherTensor = (mask*torch.ne(torch.round(y_true), torch.round(y_pred))).float()
    res = sum(sum(myOtherTensor))/(torch.sum(mask))
    return res

def errors_bler(y_true, y_pred, get_pos = False):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    if not get_pos:
        return bler_err_rate
    else:
        err_pos = list(np.nonzero((np.sum(tp0,axis=1)>0).astype(int))[0])
        return bler_err_rate, err_pos