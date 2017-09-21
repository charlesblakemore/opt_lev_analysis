###############################################################################
#storing seemingly useful functions that no longer make sense where they are

def round_sig(x, sig=2):
    '''Round a number to a certain number of sig figs
           INPUTS: x, number to be rounded
                   sig, number of sig figs

           OUTPUTS: num, rounded number'''
    neg = False
    if x == 0:
        return 0
    else:
        if x < 0:
            neg = True
            x = -1.0 * x
        num = round(x, sig-int(math.floor(math.log10(x)))-1)
        if neg:
            return -1.0 * num
        else:
            return num

