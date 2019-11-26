import numpy as np
import copy


def rand_complex_tensor(s, seed = 0):
    '''generates a random complex tensor with the given shape.'''
    np.random.seed(seed)
    return np.random.randn(*s) + 1.j*np.random.randn(*s)

def project(v, u):
    return np.vdot(u, v)*u/np.vdot(u, u)

def on_basis(init_basis):
    '''takes a template with shape (ndof, nharms) and returns a 
        (narhm, ndof, naharm) basis where the first element is the 
        templates and the other elements along the zeroth axis are
        orthogonal to the template and have the same norm as the 
        template.'''
    template = init_basis[0]
    n_temp = np.sqrt(np.einsum('i,i', template, np.conj(template)))
    st = np.shape(template)
    #perform GS orthonormalization
    for i, bv in enumerate(init_basis[1:]):
        for j in range(i+1):
            init_basis[i+1] -= project(init_basis[i+1], init_basis[j]) 
    init_basis = np.einsum('ik, i->ik', init_basis, 1./np.sqrt(np.einsum('ik,ik->i', init_basis, np.conj(init_basis))))
    return init_basis

def random_basis(threedtemplate):
    '''takes a 3d template with shape (3, nharm) and makes a random (nharm, 3, nharm) 
       basis with the template as the first element'''
    s = np.shape(threedtemplate)
    return np.concatenate([[threedtemplate], rand_complex_tensor((s[1]-1, s[0], s[1]))])

def make_3don_basis(threedtemplate):
    '''Applies make_basis for each direction of the set of templates with shape (3, nharm)'''
    init_basis = random_basis(threedtemplate)
    init_basis[:, 0, :] = on_basis(init_basis[:, 0, :])
    init_basis[:, 1, :] = on_basis(init_basis[:, 1, :])
    init_basis[:, 2, :] = on_basis(init_basis[:, 2, :])
    return init_basis

def normalize_basis(threedonbasis, threedtemplate):
    '''normalizes (nharm, 3, nharm) basis by norm of 3d template'''
    xnorm = np.sqrt(np.vdot(threedtemplate[0, :], threedtemplate[0, :]))
    ynorm = np.sqrt(np.vdot(threedtemplate[1, :], threedtemplate[1, :]))
    znorm = np.sqrt(np.vdot(threedtemplate[2, :], threedtemplate[2, :]))
    threedonbasis[:, 0, :]/=xnorm
    threedonbasis[:, 1, :]/=ynorm
    threedonbasis[:, 2, :]/=znorm
    return threedonbasis

def apply_coord_trans(threedbasis, datffts_in):
    '''applies coordinate transformation to alpha basis'''
    datffts = copy.deepcopy(datffts_in)
    for i in range(2):
        datffts[i, :] = np.dot(np.conjugate(threedbasis[:, i, :]), datffts[i, :])
    return datffts

def apply_trans_all_data(threedbasis, all_dat_ffts):
    '''applies coordinate transofrmation to all of the ffts'''
    out_ffts = np.zeros((len(all_dat_ffts), 3, 11), dtype = complex)
    for i, d in enumerate(all_dat_ffts):
        out_ffts[i, :, :] = apply_coord_trans(threedbasis, d)

    return out_ffts








