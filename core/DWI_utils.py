# -*- coding: utf-8 -*-
"""
Snippets of code from a file created on Mon Dec 17 15:53:24 2018
Author : Gaëtan Rensonnet
@author: rensonnetg
"""
import numpy as np
from scipy.interpolate import interp1d

#############################################################################
# DW-MRI
#############################################################################

def rotate_atom(sig, sch_mat, ordir, newdir, DIFF, S0, warnings=True):
    """Rotate HARDI DW-MRI signals arising from single fascicles.

    Assumes signals split per shell can be written as a function of the dot
    product between the applied gradient orientation and the main orientation
    of the fascicle of axons. Also assumes that diffusion is FREE along the
    main orientation of the fascicle.

    Args:
      sig: NumPy 1-D or 2D array of shape (Nmris,) or (Nmris, Nsub);
      sch_mat: NumPy 2-D array of shape (Nmris, 6) or (Nmris, 7);
      ordir: NumPy 1-D array of shape (3,) specifying main orientation of sig;
      newdir: NumPy 1-D array of shape (3,) specifying the new main
          orientation of the rotated signals;
      DIFF: floating-point scalar. Used to add the free diffusion data point
          (1, E_free*S0) to stabilize interpolation in interval [0, 1];
      S0: NumPy array of the same shape as sig containing the non diffusion-
          weighted signal valules. Within a substrate and a HARDI shell, all
          values should be identical. Like DIFF, just used to add the
          free-diffusion data point to stablize the interpolation.
      warnings: whether to print warning to stdout. Default: True.

    Returns:
      sig_rot: NumPy array of the same shape as sig containing the rotated
          signals(s)

    """
    # Check inputs
    assert isinstance(sig, np.ndarray), "Input sig should be a NumPy ndarray"
    assert isinstance(sch_mat, np.ndarray), ("Input sch_mat should be a "
                                             "NumPy ndarray")
    assert isinstance(ordir, np.ndarray), ("Input ordir should be a NumPy "
                                           "ndarray")
    assert isinstance(newdir, np.ndarray), ("Input newdir should be a "
                                            "NumPy ndarray")
    # Fix stupid Python way to slice 2D arrays into 1D-none arrays (x,)
    sig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig.reshape((sig.size, 1))

    if not isinstance(DIFF, np.ndarray):
        DIFF = np.array([[DIFF]])  # make 2D

    assert isinstance(S0, np.ndarray), "Input S0 should be a NumPy ndarray"
    if S0.ndim == 1:
        S0 = S0[:, np.newaxis]

    if sch_mat.shape[1] < 6:
        raise ValueError('sch_mat must be a N-by-6 or7 matrix')
    if sch_mat.shape[0] != sig.shape[0]:
        raise ValueError('sch_mat and sig must have the same number of rows')
    assert sig.shape == S0.shape, ("The S0 matrix should have the same size "
                                   "as the signal matrix")

    num_subs = sig.shape[1]
    gam = 2*np.pi*42.577480e6  # 2*np.pi*42.577480e6
    ordirnorm = np.sqrt((ordir**2).sum())
    newdirnorm = np.sqrt((newdir**2).sum())

    Gdir_norm_all = np.sqrt((sch_mat[:, 0:3]**2).sum(axis=1, keepdims=True))
    # b0 image have gradient with zero norm. Avoid true_divide warning:
    Gdir_norm_all[Gdir_norm_all == 0] = np.inf
    orcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3]/Gdir_norm_all,
                                  ordir/ordirnorm))  # Ngrad x None
    newcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3]/Gdir_norm_all,
                                   newdir/newdirnorm))  # Ngrad

#    # EDIT
#    # Sort once and for all then extract shells
#    i_orcyldotG_srt = np.argsort(orcyldotG_all)
#    GdG_un_2, i_un_2 = np.unique(sch_mat[i_orcyldotG_srt, 3:6],
#                                 return_inverse=True, axis=0)
#    # END EDIT

    # Iterate over all unique (G, Del, del) triplets
    bvals = ((gam*sch_mat[:, 3] * sch_mat[:, 5])**2
             * (sch_mat[:, 4] - sch_mat[:, 5]/3))
    sig_rot = np.zeros(sig.shape)
    GdD_un, i_un = np.unique(sch_mat[:, 3:6], return_inverse=True, axis=0)
    num_shells = GdD_un.shape[0]

    for i in range(num_shells):
        ind_sh = np.where(i_un == i)[0]  # returns a tuple
        bval = bvals[ind_sh[0]]  # Ngrad_shell

#        # EDIT
#        ind_sh_chk = i_orcyldotG_srt[i_un_2==i]
#        assert (ind_sh_chk.shape[0]==
#                ind_sh.shape[0]), ("Problem with new shell indices "
#                                   "for shell %d/%d"%(i, num_shells))
#        # END EDIT

        # No rotation for b0 images
        if bval == 0:
            sig_rot[ind_sh, :] = sig[ind_sh, :]
            continue

        # A shell should contain at least two data points
        if ind_sh.size < 2:
            raise ValueError("Fewer than 2 identical (G, Del, del) triplets "
                             "detected for triplet %d/%d (%g, %g, %g), b=%g"
                             " s/mm^2, probably not a HARDI shell." %
                             (i+1, num_shells, GdD_un[i, 0], GdD_un[i, 1],
                              GdD_un[i, 2], bval/1e6))
        # Print warning if very few points detected in one shell
        if ind_sh.size < 10 and warnings:
            print("WARNING: rotate_atom: fewer than 10 data points detected"
                  " for acquisition parameters (G, Del, del) %d/%d "
                  "(%g, %g, %g), b=%g s/mm^2.\n"
                  "Quality of approximation may be poor."
                  % (i+1, num_shells, GdD_un[i, 0], GdD_un[i, 1],
                     GdD_un[i, 2], bval/1e6))

        # Check that non diffusion weighted values are identical in a shell
        # for each substrate separately
        S0_sh_ok = np.all(np.isclose(S0[ind_sh, :],
                                     S0[ind_sh[0], :]),
                          axis=0)  # Nsubs
        if np.any(~S0_sh_ok):
            bad_subs = np.where(~S0_sh_ok)[0]
            raise ValueError('Distinct values in provided S0 image '
                             'for shell  %d/%d (b=%g s/mm^2) '
                             'for %d substrate(s) [%s]' %
                             (i+1, num_shells, bval/1e6,
                              bad_subs.shape[0],
                              " ".join("{:d}".format(b) for b in bad_subs)))

        # Double check unicity of (G,Del,del) triplets (redundant)
        Gb = np.unique(sch_mat[ind_sh, 3])
        Delb = np.unique(sch_mat[ind_sh, 4])
        delb = np.unique(sch_mat[ind_sh, 5])
        if Gb.size > 1:
            raise ValueError('Distinct G values detected (G1=%g, G2=%g, ...) '
                             'for triplet %d/%d, b-value %g s/mm^2' %
                             (Gb[0], Gb[1], i+1, num_shells, bval/1e6))
        if Delb.size > 1:
            raise ValueError('Distinct Del values detected (D1=%g, D2=%g, ...)'
                             ' for triplet %d/%d, b=%g s/mm^2' %
                             (Delb[0], Delb[1], i+1, num_shells, bval/1e6))
        if delb.size > 1:
            raise ValueError('Distinct del values detected (d1=%g, d2=%g, ...)'
                             'for triplet %d/%d, b=%g s/mm^2' %
                             (delb[0], delb[1], i+1, num_shells, bval/1e6))

        # Sort shell data as a function of dot product with original cyl dir
        # FIXME: keep average signal for identical dot products instead of just
        # keeping one data point via use of np.unique
        (sorted_orcyl_uni,
         sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                           return_index=True)
        dot_prod_data = sorted_orcyl_uni  # Ngrad_shell x None

        newcyldotG = newcyldotG_all[ind_sh]  # Ngrad_shell x None

        sig_or_shell = sig[ind_sh, :]  # Ngrad_shell x Nsubs
        sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # Ngrad_shell x Nsubs

#        # EDIT
#        # Problem: it still does not solve the problem of duplicate x values
#        # which will cause interp1 to throw an error so some sort of call to
#        # unique would still be required...
#        # Plus it's not clear whether sorting everything at the beginning
#        # really is faster than sorting smaller chunks in this loop
#        dot_prod_data_chk = orcyldotG_all[ind_sh_chk]
#        sig_data_chk = sig[ind_sh_chk, :]
#        newcyldotG_chk = newcyldotG_all[ind_sh_chk]
#        # for newcyldotG the order would be different but that does not matter
#        # as long as the receiver order in sig_rot[ind_sh_chk]
#        # matches newcyldotG_all[ind_sh_chk]
#        assert (np.all(dot_prod_data ==
#                       dot_prod_data_chk)), "Problem with dot_prod_data"
#        assert (np.all(sig_data ==
#                       sig_data_chk)), ("Problem with sig_data for "
#                                        "shell %d/%d" % (i, num_shells))
#
#        # END EDIT

        # Add the data point (1, E_free*S0_b) to better span the interpolation
        # interval :
        if not np.any(dot_prod_data == 1):
            dot_prod_data = np.append(dot_prod_data, [1])  # Ngrad_sh+1 x 1
            if num_subs == 1:
                sig_data = np.append(sig_data,
                                     np.exp(-bval * DIFF) * S0[ind_sh[0]])
                sig_data = sig_data[:, np.newaxis]
            else:
                # DIFF is 1x1 ndarray so np.exp() is too and free diff is 2D
                free_diff = np.exp(-bval * DIFF) * S0[ind_sh[0], :]
                sig_data = np.append(sig_data,  # Ngrad_sh+1 x Nsubs
                                     free_diff, axis=0)

        # Smooth out data near dotproduct ~= 0, to avoid extrapolation
        # unstabilities due to MC variability with closely-spaced data :
        almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
        cluster_size = np.sum(almost_perp)

        # Subsitute the x values close to left-most edge by their center of
        # mass. The if statement avoids taking the mean of an empty array,
        # which returns nan in NumPy (!). Do the same with the measurements.
        if cluster_size > 1:
            dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                      dot_prod_data[cluster_size:])
            # has shape (Ngrad_sh-Nzeros) x None
            sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                         axis=0,
                                         keepdims=True),
                                 sig_data[cluster_size:, :],
                                 axis=0)

        # Check consistency of interpolation data
        if (dot_prod_data.size != sig_data.shape[0]) and warnings:
            print("WARNING: rotate_atom: problem with shapes and/or sizes"
                  " before 1D interpolation at shell %d/%d "
                  "(G=%g Del=%g del=%g)" %
                  (i+1, num_shells,
                   GdD_un[i, 0], GdD_un[i, 1], GdD_un[i, 2]))

        # Apply rotation to each substrate for the current shell in batch mode
        f_interp_shell = interp1d(dot_prod_data, sig_data, axis=0,
                                  kind='linear', fill_value='extrapolate',
                                  assume_sorted=True)
        sig_rot[ind_sh, :] = f_interp_shell(newcyldotG)

        if np.any(np.isnan(sig_rot[ind_sh, :])):
            sub_has_nan = np.any(np.isnan(sig_rot[ind_sh, :]), axis=0)
            bad_subs = np.where(sub_has_nan)[0]
            raise ValueError('Nan detected after rotation of substrate(s) '
                             'for sequence(s) {%d...%d} (bval=%g s/mm^2) '
                             'for %d substrate(s): [%s]' %
                             (ind_sh[0], ind_sh[-1], bval/1e6,
                              bad_subs.shape[0],
                              " ".join("%d" % b for b in bad_subs)))
    return np.reshape(sig_rot, sig_shape)



def interp_PGSE_from_multishell(sch_mat, newdir,
                                sig_ms=None, sch_mat_ms=None, ordir=None,
                                msinterp=None):
    """Single-fascicle PGSE signal interpolated from dense multi-HARDI.

    This script can also be used to perform rotations based on the dense
    sampling rather than on the dictionary along the canonical direction.

    Args:
      sch_mat: 2D NumPy array with shape (Nseq, 7)
      newdir: NumPy array with 3 elements having unit Euclidean norm.
      sig_ms: 2D NumPy array with shape (Nsampling, Nsubs) or a 1D array with
        shape (Nsampling,).
      sch_mat_ms: 2D NumPy array with shape (Nsampling, 7)
      ordir: NumPy array with 3 elements having unit Euclidean norm.
      msinterp: interpolator object such as returned by
        init_PGSE_multishell_interp. If set, sig_ms, sch_mat_ms and ordir need
        not be set (they will be ignored if provided) and only sch_mat and
        newdir are required.

    Returns:
      A 2D NumPy array with shape (Nseq, Nsub) passed to NumPy.squeeze(),
      i.e., the resulting array could be 1D (Nseq,) or even a scalar if Nseq
      and Nsub are 1.

    """
    FAST_MODE = msinterp is not None

    if FAST_MODE:  # initialized mode
        if (msinterp['Gms_un'].size !=
                len(msinterp['interpolators'])):
            raise ValueError("msinterp['Gms_un'] has size %d vs "
                             "expected %d to match "
                             "len(msinterp['shell_interpolators'])"
                             % (msinterp['Gms_un'].size,
                                len(msinterp['shell_interpolators'])))
        # Check interpolator object
        num_subs_chk = msinterp['interpolators'][0].y.shape[1]
        if num_subs_chk != msinterp['num_subs']:
            raise ValueError("Inconsistency in msinterp "
                             "regarding number of substrates. "
                             "Make sure the interpolator was initialized"
                             " on the right dictionary.")
        num_subs = msinterp['num_subs']
        sch_DeldelTE = msinterp['scheme_DeldelTE']
    else:  # non-initialized mode
        msinterp = {}  # must be initialized
        if sig_ms is None or sch_mat_ms is None or ordir is None:
            raise ValueError("If msinterp is not specified, sig_ms, "
                             "sch_mat_ms and ordir must all be specified.")
        # Check gradient norm of dense multishell scheme
        Gdir_norm_ms = np.sqrt(np.sum(sch_mat_ms[:, 0:3]**2, axis=1))
        if np.any(np.abs(1 - Gdir_norm_ms[Gdir_norm_ms > 0]) > 1e-3):
            raise ValueError("Gradient directions in multi-shell scheme"
                             " matrix should all either have zero "
                             "or unit norm.")
        if ordir.size != 3:
            raise ValueError("Direction of dictionary computed with dense"
                             " sampling (ordir) should have 3 entries.")
        if ordir.ndim > 1:
            # Get rid of singleton dimensions because calls to 1D
            # interpolators below will be affected and dimension
            # mismatch will occur
            ordir = np.squeeze(ordir)
        # Check consistency of the timing parameters in the acquisitions
        sch_DeldelTE = sch_mat_ms[0, 4:7]
        chk_ms = np.isclose(sch_DeldelTE, sch_mat_ms[:, 4:7])
        if not np.all(chk_ms):
            raise ValueError("Delta, delta and TE values should all be "
                             "identical in multi-shell sampling.")
        # Get inputs in right format
        if sig_ms.ndim == 1:
            sig_ms = sig_ms.reshape((sig_ms.size, 1))
        if sch_mat_ms.shape[0] != sig_ms.shape[0]:
            raise ValueError("Number of lines in dense multishell scheme"
                             " (%d) does not match number of signal values"
                             " per substrate (%d)." %
                             (sch_mat_ms.shape[0], sig_ms.shape[0]))
        ordirnorm = np.sqrt((ordir**2).sum())  # (1,)
        if np.abs(1 - ordirnorm) > 1e-3:
            raise ValueError("Orientation vector of the multi-shell signal"
                             " must have unit norm. Detected"
                             " %g." % (ordirnorm,))
        num_subs = sig_ms.shape[1]
        # Dot-products between fascicle directions and gradient directions
        orcyldotG_all = np.abs(np.dot(sch_mat_ms[:, 0:3], ordir))  # (Nmsh,)
        # Find unique G values in dense sampling
        msinterp['Gms_un'], i_Gms = np.unique(sch_mat_ms[:, 3],
                                              return_inverse=True)
    # At this point, we have initialized:
    #  num_subs, sch_DeldelTE, msinterp['Gms_un'], i_Gms, ordir, newdir

    # W/ or W/o initialization: check inputs
    chk_new = np.isclose(sch_DeldelTE, sch_mat[:, 4:7])
    if not np.all(chk_new):
        raise ValueError("Delta, delta and TE values should all be "
                         "identical to those in the multi-shell sampling.")
    if newdir.size != 3:
        raise ValueError("Direction of fascicle for new signal (newdir)"
                         " should have 3 entries.")
    if newdir.ndim > 1:
        # Get rid of singleton dimensions because calls to 1D
        # interpolators below will be affected and dimension
        # mismatch will occur
        newdir = np.squeeze(newdir)
    newdirnorm = np.sqrt((newdir**2).sum())  # (1,)
    if np.abs(1 - newdirnorm) > 1e-3:
        raise ValueError("Orientation vector "
                         "of the new signal must have unit norm. Detected"
                         " %g." % (newdirnorm,))
    # Check gradient norms of new acquisition scheme
    Gdir_norm_new = np.sqrt(np.sum(sch_mat[:, 0:3]**2, axis=1))
    if np.any(np.abs(1 - Gdir_norm_new[Gdir_norm_new > 0]) > 1e-3):
        raise ValueError("Gradient directions in multi-shell scheme matrix"
                         " should all either have zero or unit norm.")

    # Dot-products between new fascicle and new gradient directions
    newcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3], newdir))  # (Nmri,)
    # Find unique G values in new scheme
    G_un, i_G = np.unique(sch_mat[:, 3], return_inverse=True)

    # Establish mapping: which G in multi-shell samping to use for
    # each new G?
    Gnew_to_Gms = []
    Gms_used = np.full(msinterp['Gms_un'].shape,
                       False,
                       dtype=bool)  # just to speed up non-initialized mode
    for Gnew in G_un:
        # Check for precisely same value in multi-shell sampling
        i = np.where(Gnew == msinterp['Gms_un'])[0]
        if i.size > 0:
            Gnew_to_Gms.append(i)  # i is np.array of shape (1,)
            Gms_used[i] = True
        else:
            # No identical G found, find surrounding values in mshell
            # sampling
            i_high = np.argmax(msinterp['Gms_un'] > Gnew)
            if i_high == 0:
                raise ValueError("Gradient intensity %g is not in the [%g, %g]"
                                 " range spanned by the multi-shell sampling."
                                 " Extrapolation not supported." %
                                 (Gnew,
                                  msinterp['Gms_un'][0],
                                  msinterp['Gms_un'][-1]))
            Gnew_to_Gms.append([i_high - 1, i_high])
            Gms_used[i_high - 1] = True
            Gms_used[i_high] = True

    if not FAST_MODE:
        # Compute spherical interpolator for each shell of the dense sampling
        # which will be used for the new signals
        msinterp['interpolators'] = []
        for i in range(msinterp['Gms_un'].shape[0]):
            if not Gms_used[i]:
                # Skip if it's not even going to be used in the interpolation
                msinterp['interpolators'].append(None)
                continue

            # Get lines of the multi-shell dense sampling
            ind_sh = np.where(i_Gms == i)[0]  # (Ngr_sh,)

            # If G==0, check that all G=0 signals identical in each column
            # and do not compute a spherical interpolator because it does
            # not make sense
            if msinterp['Gms_un'][i] == 0:
                chk = np.all(np.isclose(sig_ms[ind_sh, :],
                                        sig_ms[ind_sh[0], :]),
                             axis=0)  # (Nsubs,)
                if np.any(~chk):
                    bad_subs = np.where(~chk)[0]
                    raise ValueError('Distinct signal values in provided '
                                     'multi-shell sampling for zero gradients,'
                                     ' for %d substrate(s) [%s]' %
                                     (bad_subs.shape[0],
                                      " ".join("{:d}".format(b)
                                               for b in bad_subs)))
                # For b=0 (G=0) shells, an interpolator is just a constant
                # function. We just repeat, for each substrate, the first
                # signal value twice (in case only one b=0 acquisition present
                # in dense multi-shell sampling, which is likely for any
                # efficient sampling) for the x-axis points 0 and 1, the
                # extreme values of the dot product
                f_interp_shell = interp1d([0, 1],  # data below is (2, Nsubs)
                                          np.repeat([sig_ms[ind_sh[0], :]],
                                                    2, axis=0),
                                          axis=0,
                                          kind='linear',
                                          fill_value='extrapolate',
                                          assume_sorted=True)
                msinterp['interpolators'].append(f_interp_shell)
                continue
            # General G > 0 case in dense multi-shell sampling
            (sorted_orcyl_uni,
             sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                               return_index=True)
            dot_prod_data = sorted_orcyl_uni  # (Ngr_sh,)
            sig_or_shell = sig_ms[ind_sh, :]  # (Ngr_sh, Nsubs)
            sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # (Ngr_sh, Nsubs)

            # TODO: Add the data point (1, E_free*S0_b)

            # Smooth out data near dotproduct ~= 0, to avoid extrapolation
            # unstabilities due to MC variability with closely-spaced data :
            almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
            cluster_size = np.sum(almost_perp)

            # Substitute the x values close to left-most edge by their
            # center of mass. The if statement avoids taking the mean of an
            # empty array, which returns nan in Numpy (!). Do the same with
            # the measurements.
            if cluster_size > 1:
                dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                          dot_prod_data[cluster_size:])
                sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                             axis=0,
                                             keepdims=True),
                                     sig_data[cluster_size:, :],
                                     axis=0)  # ((Ngr_sh-Nzeros), Nsubs)
            # Compute spherical interpolator of shell
            f_interp_shell = interp1d(dot_prod_data,
                                      sig_data,
                                      axis=0,
                                      kind='linear',
                                      fill_value='extrapolate',
                                      assume_sorted=True)
            msinterp['interpolators'].append(f_interp_shell)
    # END of if not FAST_MODE

    # Perform interpolation inside thick spherical shells
    # First on the shells of the multi-shell sampling, then along G between
    # shells
    sig_new = np.zeros((sch_mat.shape[0], num_subs))
    for i in range(G_un.shape[0]):
        # Lines of new protocol
        ind_sh = np.where(i_G == i)[0]
        newcyldotG = newcyldotG_all[ind_sh]  # (Ngrad_shell,)

        # Identical G: just a rotation, i.e. 1D interpolation
        if len(Gnew_to_Gms[i]) == 1:
            i_int = Gnew_to_Gms[i][0]
            sig_new[ind_sh,
                    :] = msinterp['interpolators'][i_int](newcyldotG)
            continue

        # Case where multiple reference shells are needed for interpolation
        Gms_ref = msinterp['Gms_un'][Gnew_to_Gms[i]]
        sig_ref_shells = np.zeros((Gms_ref.shape[0],
                                   newcyldotG.shape[0],
                                   num_subs))
        for j in range(Gms_ref.shape[0]):
            # First, interpolate at every shell of the multi-shell scheme
            i_int = Gnew_to_Gms[i][j]
            sig_ref_shells[j,
                           :,
                           :] = msinterp['interpolators'][i_int](newcyldotG)
        # Interpolate for new G value between shells, simultaneously for all
        # new directions:
        f_interp_G = interp1d(Gms_ref, sig_ref_shells,
                              axis=0,
                              kind='linear',
                              fill_value=np.nan,
                              assume_sorted=True)
        sig_new[ind_sh, :] = f_interp_G(G_un[i])
    return np.squeeze(sig_new)  # squeeze if num_subs=1


def init_PGSE_multishell_interp(sig_ms, sch_mat_ms, ordir):
    """Initializes multi-shell interpolator for faster interpolation.

    The output can be provided to interp_PGSE_from_multishell to make
    interpolation faster.

    Args:
      sig_ms: 2D NumPy array with shape (Nsampling, Nsubs) or a 1D array with
        shape (Nsampling,).
      sch_mat_ms: 2D NumPy array with shape (Nsampling, 7)
      ordir: NumPy array with 3 elements having unit Euclidean norm.

    Returns:
      interpolator object with precomputed settings which will make calls to
        interp_PGSE_from_multishell faster.
    """
    if ordir.size != 3:
        raise ValueError("Direction of dictionary computed with dense"
                         " sampling (ordir) should have 3 entries.")
    if ordir.ndim > 1:
        # Get rid of singleton dimensions because calls to 1D interpolators
        # below will be affected and dimension mismatch will occur
        ordir = np.squeeze(ordir)
    # Check inputs
    chk_ms = np.isclose(sch_mat_ms[0, 4:7], sch_mat_ms[:, 4:7])
    if not np.all(chk_ms):
        raise ValueError("Delta, delta and TE values should all be "
                         "identical in multi-shell sampling.")

    # Get inputs in right format
    if sig_ms.ndim == 1:
        sig_ms = sig_ms.reshape((sig_ms.size, 1))

    ordirnorm = np.sqrt((ordir**2).sum())  # (1,)
    if np.abs(1 - ordirnorm) > 1e-3:
        raise ValueError("Orientation vector of the multi-shell signal "
                         "must have unit norm. Detected"
                         " %g and %g respectively." % (ordirnorm,))
    # Check gradient norms of both schemes
    Gdir_norm_ms = np.sqrt(np.sum(sch_mat_ms[:, 0:3]**2, axis=1))
    if np.any(np.abs(1 - Gdir_norm_ms[Gdir_norm_ms > 0]) > 1e-3):
        raise ValueError("Gradient directions in multi-shell scheme matrix"
                         " should all either have zero or unit norm.")

    num_subs = sig_ms.shape[1]

    # Dot-products between fascicle directions and gradient directions
    orcyldotG_all = np.abs(np.dot(sch_mat_ms[:, 0:3], ordir))  # (Nmsh,)
    # Find unique G values
    Gms_un, i_Gms = np.unique(sch_mat_ms[:, 3], return_inverse=True)

    # Compute spherical interpolator for each shell of the dense sampling which
    # will be used for the new signals
    interpolators = []
    for i in range(Gms_un.shape[0]):
        # Get lines of the multi-shell dense sampling
        ind_sh = np.where(i_Gms == i)[0]  # (Ngrad_sh,)

        # If G==0, check that all G=0 signals identical in each column and do
        # not compute a spherical interpolator because it does not make sense
        if Gms_un[i] == 0:
            chk = np.all(np.isclose(sig_ms[ind_sh, :],
                                    sig_ms[ind_sh[0], :]),
                         axis=0)  # (Nsubs,)
            if np.any(~chk):
                bad_subs = np.where(~chk)[0]
                raise ValueError('Distinct signal values in provided multi-'
                                 'shell sampling for zero gradients '
                                 '(b0 acquistions), for '
                                 '%d substrate(s) [%s]' %
                                 (bad_subs.shape[0],
                                  " ".join("{:d}".format(b)
                                           for b in bad_subs)))
            # For b=0 (G=0) shells, an interpolator is just a constant
            # function. We just repeat, for each substrate, the first signal
            # value twice (in case only one b=0 acquisition present in
            # dense multi-shell sampling, which is likely for any efficient
            # sampling) for the x-axis points 0 and 1, the extreme values of
            # the dot product
            f_interp_shell = interp1d([0, 1],  # data below is (2, Nsubs) array
                                      np.repeat([sig_ms[ind_sh[0], :]],
                                                2, axis=0),
                                      axis=0,
                                      kind='linear',
                                      fill_value='extrapolate',
                                      assume_sorted=True)
            interpolators.append(f_interp_shell)
            continue
        # General G > 0 case in dense multi-shell sampling
        (sorted_orcyl_uni,
         sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                           return_index=True)
        dot_prod_data = sorted_orcyl_uni  # (Ngrad_sh,)
        sig_or_shell = sig_ms[ind_sh, :]  # (Ngrad_sh, Nsubs)
        sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # (Ngrad_sh, Nsubs)

        # TODO: Add the data point (1, E_free*S0_b)

        # Smooth out data near dotproduct ~= 0, to avoid extrapolation
        # unstabilities due to MC variability with closely-spaced data :
        almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
        cluster_size = np.sum(almost_perp)

        # Substitute the x values close to left-most edge by their center of
        # mass. The if statement avoids taking the mean of an empty array,
        # which returns nan in Numpy (!). Do the same with the measurements.
        if cluster_size > 1:
            dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                      dot_prod_data[cluster_size:])
            sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                         axis=0,
                                         keepdims=True),
                                 sig_data[cluster_size:, :],
                                 axis=0)  # ((Ngrad_sh-Nzeros), Nsubs)
        # Compute spherical interpolator of shell
        f_interp_shell = interp1d(dot_prod_data,
                                  sig_data,
                                  axis=0,
                                  kind='linear',
                                  fill_value='extrapolate',
                                  assume_sorted=True)
        interpolators.append(f_interp_shell)
    output = {'scheme_DeldelTE': sch_mat_ms[0, 4:7],
              'num_subs': num_subs,
              'Gms_un': Gms_un,
              'interpolators': interpolators}
    return output


def project_PGSE_scheme_xy_plane(sch_mat):
    """Removes gradients' component along z-axis

    Args:
      sch_mat: 2D NumPy array or string path to a text file with a one-line
        header. The first four entries of each row should be
        [gx, gy, gz, G] where [gx, gy, gz] is a unit-norm vector.

    Returns:
      The scheme matrix projected in the xy plane, i.e. with gz'=0, [gx', gy']
      having unit norm and G' such that (gz*G)**2 + G'**2 = G**2.
    """
    if isinstance(sch_mat, str):
        schemefile = sch_mat
        sch_mat = np.loadtxt(schemefile, skiprows=1)
    if sch_mat.ndim == 1:
        sch_mat = sch_mat[np.newaxis, :]
    # Proportion of gradient intensitiy in xy plane
    gxy = np.sqrt(sch_mat[:, 0]**2 + sch_mat[:, 1]**2)

    sch_mat_xy = np.zeros(sch_mat.shape)
    # Adjust gradient magnitudes in xy plane
    sch_mat_xy[:, 3] = sch_mat[:, 3] * gxy
    gxy[gxy == 0] = 1  # numerical stability before dividing
    # Normalize gradient directions to unit norm
    sch_mat_xy[:, :2] = sch_mat[:, :2]/gxy[:, np.newaxis]
    # Make sure zero gradients remain zero
    sch_mat_xy[sch_mat[:, 3] == 0, :4] = 0
    # Rest of protocol parameters remain unchanged (Del, del, TE)
    sch_mat_xy[:, 4:] = sch_mat[:, 4:]

    Gz = np.abs(sch_mat[:, 2]) * sch_mat[:, 3]
    G_chk_sq = sch_mat_xy[:, 3]**2 + Gz**2
    msg = ("Inconsistency with gradient intensities during"
           " projection in xy plane")
    assert np.all(np.abs(np.sqrt(G_chk_sq) - sch_mat[:, 3])
                  <= 1e-4 * sch_mat[:, 3]), msg  # keep <= for zeros !
    return sch_mat_xy


def import_PGSE_scheme(scheme):
    """Import PGSE scheme file or matrix

    Args:
      scheme: path (str) to scheme file or NumPy array containing 7 entries
        per row.
        Each row is of the form [gx, gy, gz, G, Del, del, TE] where
        [gx, gy, gz]^T is a the gradient direction with unit Euclidean norm,
        G is the gradient intensity, Del the time between the onsets of the
        two gradient pulses and del the duration of each gradient pulse. The
        values must satisfy Del>=del and TE>=(Del+del).

    Returns:
      Always a 2D NumPy array with 7 entries per row.
    """
    if isinstance(scheme, str):
        # Load from text file
        with open(scheme, 'r') as f:
            # Get header
            first_line = f.readline()
        rows_to_skip = 0
        if 'version' in first_line.lower():
            rows_to_skip = 1
        sch_mat = np.loadtxt(scheme, skiprows=rows_to_skip)
    elif isinstance(scheme, np.ndarray):
        sch_mat = scheme
    else:
        raise TypeError("Unable to import a PGSE scheme matrix from input")
    if sch_mat.ndim == 1:
        # Return a 2D row matrix if only one sequence in protocol
        sch_mat = sch_mat[np.newaxis, :]
    if sch_mat.shape[1] != 7:
        raise RuntimeError("Detected %s instead of expected 7 colums in"
                           " PGSE scheme matrix." % sch_mat.shape[1])
    grad_norm = np.sqrt(np.sum(sch_mat[:, :3]**2, axis=1))
    num_bad_norms = np.sum(np.abs(1-grad_norm[grad_norm > 0]) > 1e-4)
    if num_bad_norms > 0:
        raise ValueError("Detected %d non-zero gradients which did not have"
                         " unit norm. Please normalize." % num_bad_norms)
    G = sch_mat[:, 3]
    Delta = sch_mat[:, 4]
    delta = sch_mat[:, 5]
    TE = sch_mat[:, 6]
    if np.any(G < 0):
        raise ValueError('Detected %d sequence(s) with negative gradient '
                         'intensity (4th column).' % np.sum(G < 0))
    if np.any(Delta < 0):
        raise ValueError('Detected %d sequence(s) with negative gradient '
                         'separation Delta (5th column).' % np.sum(Delta < 0))
    if np.any(delta < 0):
        raise ValueError('Detected %d sequence(s) with negative gradient '
                         'duration delta (6th column).' % np.sum(delta < 0))
    if np.any(TE < 0):
        raise ValueError('Detected %d sequence(s) with negative echo time '
                         'TE (7th column).' % np.sum(TE < 0))
    if np.any(delta > Delta):
        raise ValueError('Detected %d sequence(s) in which delta (6th column)'
                         ' was greater than Delta (5th column).' %
                         np.sum(delta > Delta))
    if np.any(TE < (Delta+delta)*0.999):
        # this comparison is subject to numerical round-off errors
        raise ValueError('Detected %d sequence(s) in which TE (7th column)'
                         ' was lower than Delta+delta.' %
                         np.sum(TE < (Delta+delta)))
    return sch_mat



def gen_SoS_MRI(S0, sigma_g, N=1):
    """Simulates Sum-of-Squares MRI signal for phased-array systems.

    Produces S_out = sqrt{ sum_{i=1}^N |S_i|^2 },
      where S_i = S_0 + eps1 + (1i)*eps2,
          with eps1, eps2 two independent zero-mean Gaussian variables of
          standard deviation sigma_g, assumed identical in all N coils, in
          both channels, and 1i the imaginary number.
    S_out follows a non-central Chi distribution.

    Args:
      S0: N-D NumPy array representing the true, possibly complex-valued,
        MRI contrast. Its entries represent acquisition parameters and/or
        multiple voxels.
      sigma_g: scalar or N-D NumPy array. Standard deviation of the Gaussian
        white noise in each coil, always assumed identical for all N coils
        and without inter-coil correlation.
        If `sigma_g` is a scalar, the standard deviation of the Gaussian noise
        is identical for all entries of `S0`.
        Else if `sigma_g.shape` is equal to `S0.shape`, then the standard
        deviation of the noise can be different for each entry of `S0`.
      N: the effective number of coils. Default is 1 (Rician noise).

    Returns:
      A scalar or NumPy array with the same shape as S0. The noise
      realizations are completely independent from one another.

    Raises:
      ValueError: if sigma_g is not a scalar but its shape does not match that
        of S0.
    """
    if np.all(sigma_g == 0):
        return np.sqrt(N)*S0  # perfect noiseless scenario

    if (np.ndim(sigma_g) > 0 and  # sigma_g is an array
            sigma_g.size > 1 and  # not a scalar
            S0.shape != sigma_g.shape):
        raise ValueError('sigma_g should either be a scalar or have '
                         'the shape (%s) of S0 for 1-to-1 '
                         'correspondance. Detected (%s) instead.'
                         % (", ".join("%d" % s for s in S0.shape),
                            ", ".join("%d" % s for s in sigma_g.shape)))

    Y = np.zeros(S0.shape, dtype=np.float64)
    for _ in range(N):
        noise_in_phase = sigma_g*np.random.randn(*S0.shape)
        noise_in_quadrature = sigma_g*np.random.randn(*S0.shape)
        Y = Y + (S0 + noise_in_phase)**2 + noise_in_quadrature**2
    # Pathological case when S0 has shape (N,) and sigma_g has shape (1, 1)
    # because due to NumPy's broadcasting rules Y will have shape (1, N)
    # instead of the desired (N,).
    return np.reshape(np.sqrt(Y), S0.shape)



