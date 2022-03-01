import math
import numpy as np
def DWA(loss_t,loss_tt,if_electra=False,T=20):
    l1_t = loss_t[0]
    l2_t = loss_t[1]
    l3_t = loss_t[2]

    l1_tt = loss_tt[0]
    l2_tt = loss_tt[1]
    l3_tt = loss_tt[2]

    if if_electra == True:
        r1 = l1_t / l1_tt
        r2 = l2_t / l2_tt
        r3 = l3_t / l3_tt
        devider = math.exp(r1/T) + math.exp(r2/T) + math.exp(r3/T)
        wa = math.exp(r1/T) / devider
        wb = math.exp(r2/T) / devider
        wc = math.exp(r3/T) / devider
        return wa,wb,wc
    else:
        r1 = l1_t / l1_tt
        r2 = l2_t / l2_tt
        devider = math.exp(r1 / T) + math.exp(r2 / T)
        wa = math.exp(r1 / T) / devider
        wb = math.exp(r2 / T) / devider
        wc=0
        return wa,wb,wc