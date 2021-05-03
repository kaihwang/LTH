from LTH import *
from nilearn.input_data import NiftiLabelsMasker
################################
## Calculate thalamocortical FC and thalamic vox by vox PC
################################

def generate_correlation_mat(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pcorr_subcortico_cortical_connectivity(subcortical_ts, cortical_ts):
	''' function to do partial correlation bewteen subcortical and cortical ROI timeseries.
	Cortical signals (not subcortical) will be removed from subcortical and cortical ROIs,
	and then pairwise correlation will be calculated bewteen subcortical and cortical ROIs
	(but not between subcortico-subcortical or cortico-cortical ROIs).
	This partial correlation/regression approach is for cleaning subcortico-cortical
	conectivity, which seems to be heavily influenced by a global noise.
	usage: pcorr_mat = pcorr_subcortico-cortical(subcortical_ts, cortical_ts)
	----
	Parameters
	----
	subcortical_ts: txt file of timeseries data from subcortical ROIs/voxels, each row is an ROI
	cortical_ts: txt file of timeseries data from cortical ROIs, each row is an ROI
	pcorr_mat: output partial correlation matrix
	'''
	from scipy import stats, linalg
	from sklearn.decomposition import PCA

	# # transpose so that column is ROI, this is because output from 3dNetcorr is row-based.
	# subcortical_ts = subcortical_ts.T
	# cortical_ts = cortical_ts.T
	cortical_ts[np.isnan(cortical_ts)]=0
	subcortical_ts[np.isnan(subcortical_ts)]=0

	# check length of data
	assert cortical_ts.shape[0] == subcortical_ts.shape[0]
	num_vol = cortical_ts.shape[0]

	#first check that the dimension is appropriate
	num_cort = cortical_ts.shape[1]
	num_subcor = subcortical_ts.shape[1]
	num_total = num_cort + num_subcor

	#maximum number of regressors that we can use
	max_num_components = int(num_vol/20)
	if max_num_components > num_cort:
		max_num_components = num_cort-1

	pcorr_mat = np.zeros((num_total, num_total), dtype=np.float)

	for j in range(num_cort):
		k = np.ones(num_cort, dtype=np.bool)
		k[j] = False

		#use PCA to reduce cortical data dimensionality
		pca = PCA(n_components=max_num_components)
		pca.fit(cortical_ts[:,k])
		reduced_cortical_ts = pca.fit_transform(cortical_ts[:,k])

		#print("Amount of varaince explanined after PCA: %s" %np.sum(pca.explained_variance_ratio_))

		# fit cortical signal to cortical ROI TS, get betas
		beta_cortical = linalg.lstsq(reduced_cortical_ts, cortical_ts[:,j])[0]

		#get residuals
		res_cortical = cortical_ts[:, j] - reduced_cortical_ts.dot(beta_cortical)

		for i in range(num_subcor):
			# fit cortical signal to subcortical ROI TS, get betas
			beta_subcortical = linalg.lstsq(reduced_cortical_ts, subcortical_ts[:,i])[0]

			#get residuals
			res_subcortical = subcortical_ts[:, i] - reduced_cortical_ts.dot(beta_subcortical)

			#partial correlation
			pcorr_mat[i+num_cort, j] = stats.pearsonr(res_cortical, res_subcortical)[0]
			pcorr_mat[j,i+num_cort ] = pcorr_mat[i+num_cort, j]

	return pcorr_mat


### global variables, masks, files, etc.
# load files
MGH_fn = '/home/kahwang/bsh/MGH/MGH/*/MNINonLinear/rfMRI_REST_ncsreg.nii.gz'
MGH_files = glob.glob(MGH_fn)

NKI_fn = '/data/backed_up/shared/NKI/*/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz'
NKI_files = glob.glob(NKI_fn)
datafiles = [NKI_files, MGH_files]
datasets=['NKI', 'MGH']

# load masks
thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_fdata()
thalamus_mask_data = thalamus_mask_data>0
thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
mm_unique = nib.load('images/mm_unique.nii.gz')
mm_unique_2mm = resample_to_img(mm_unique, thalamus_mask, interpolation = 'nearest')
Schaefer400_mask = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
cortex_masker = NiftiLabelsMasker(labels_img='/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz', standardize=False)
Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')


def cal_PC():
    '''Calculate PC values '''
    for i, files in enumerate(datafiles):

        thresholds = [95,96,97,98,99]

        # saving both patial corr and full corr
        fpc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files), len(thresholds)))
        ppc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files), len(thresholds)))
        pc_vectors = [ppc_vectors, fpc_vectors]

        for ix, f in enumerate(files):
            functional_data = nib.load(f)
            #extract cortical ts from schaeffer 400 ROIs
            cortex_ts = cortex_masker.fit_transform(functional_data)
            #time by ROI
            #cortex_ts = cortex_ts.T
            #extract thalamus vox by vox ts
            thalamus_ts = masking.apply_mask(functional_data, thalamus_mask)
            # time by vox
            #thalamus_ts = thalamus_ts.T

            # concate, cortex + thalamus voxel, dimesnion should be 2627 (400 cortical ROIs plus 2227 thalamus voxel from morel atlas)
            # work on partial corr.
            #ts = np.concatenate((cortex_ts, thalamus_ts), axis=1)
            #corrmat = np.corrcoef(ts.T)
            pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
            thalamocortical_pfc = pmat[400:, 0:400]
            #extrat the thalamus by cortex FC matrix

            # fc marices
            thalamocortical_ffc = generate_correlation_mat(thalamus_ts.T, cortex_ts.T)
            #fcmats.append(thalamocortical_fc)
            #calculate PC with the extracted thalamocortical FC matrix

            #loop through threshold
            FCmats = [thalamocortical_pfc, thalamocortical_ffc]

            for j, thalamocortical_fc in enumerate(FCmats):
                for it, t in enumerate(thresholds):
                    temp_mat = thalamocortical_fc.copy()
                    temp_mat[temp_mat<np.percentile(temp_mat, t)] = 0
                    fc_sum = np.sum(temp_mat, axis=1)
                    kis = np.zeros(np.shape(fc_sum))

                    for ci in np.unique(Schaeffer_CI):
                        kis = kis + np.square(np.sum(temp_mat[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)

                    pc_vectors[j][:,ix, it] = 1-kis

        fn = "data/%s_pc_vectors_pcorr" %datasets[i]
        np.save(fn, pc_vectors[0])
        fn = "data/%s_pc_vectors_corr" %datasets[i]
        np.save(fn, pc_vectors[1])

def cal_mmmask_FC():
    ''' calculate FC between cortical ROIs and thalamic mask of multitask impairment()'''

    for i, files in enumerate(datafiles):
        # save both pcorr and full corr
        fcpmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),400, len(files)))
        fcmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),400, len(files)))

        for ix, f in enumerate(files):
            functional_data = nib.load(f)
            #extract cortical ts from schaeffer 400 ROIs
            cortex_ts = cortex_masker.fit_transform(functional_data)
            #time by ROI
            #cortex_ts = cortex_ts.T
            #extract thalamus vox by vox ts
            thalamus_ts = masking.apply_mask(functional_data, mm_unique_2mm)
            # time by vox
            #thalamus_ts = thalamus_ts.T
            pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
            fcpmat[:,:,ix] = pmat[400:, 0:400]
            fcmat[:,:,ix] = generate_correlation_mat(thalamus_ts.T, cortex_ts.T)


        fn = "data/%s_mmmask_fc_pcorr" %datasets[i]
        np.save(fn, fcpmat)
        fn = "data/%s_mmmask_fc_fcorr" %datasets[i]
        np.save(fn, fcmat)

def cal_term_FC():
    ''' calculate FC between neurosynth term ROIs and thalamic mask of multitask impairment()'''

    terms = ['executive', 'naming', 'fluency', 'recall', 'recognition']
    nsnii={}
    for term in terms:
        fn = 'data/%s_association-test_z_FDR_0.01.nii.gz' %term
        nsnii[term] = nib.load(fn)


    for i, files in enumerate(datafiles):
        # save both pcorr and full corr
        fcpmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),len(terms), len(files)))
        fcmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),len(terms), len(files)))

        for ix, f in enumerate(files):
            functional_data = nib.load(f)

            #extract cortical ts from neurosytn priors
            cortex_ts = np.zeros((functional_data.get_fdata().shape[3], len(terms)))
            for it, term in enumerate(terms):
                ts = functional_data.get_fdata()[nsnii[term].get_fdata()>0]  #vox by time
                cortex_ts[:,it] = np.mean(ts, axis = 0) # time by term

            #cortex_ts = cortex_masker.fit_transform(functional_data)
            #time by ROI
            #cortex_ts = cortex_ts.T
            #extract thalamus vox by vox ts
            thalamus_ts = masking.apply_mask(functional_data, mm_unique_2mm)
            # time by vox
            #thalamus_ts = thalamus_ts.T
            #pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
            #fcpmat[:,:,ix] = pmat[400:, 0:400]
            fcmat[:,:,ix] = generate_correlation_mat(thalamus_ts.T, cortex_ts.T)

        fn = "data/%s_term_fc_fcorr" %datasets[i]
        np.save(fn, fcmat)

if __name__ == "__main__":

    cal_PC()
    cal_mmmask_FC()
    cal_term_FC()










#end of line
