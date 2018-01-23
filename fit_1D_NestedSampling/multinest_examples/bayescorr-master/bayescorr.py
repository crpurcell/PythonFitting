#!/usr/bin/env python

# Copyright James R Allison 2018

# Import standard and third party modules
import sys
import os
import numpy as np
from scipy import stats
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['serif'],'size':25})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#sys.path.append(os.environ['PYMULTINEST'])
import pymultinest

# Define prior function
def prior_call(types, pmins, pmaxs):
	def prior(cube, ndim, nparams):
		for i in range(ndim):
			if types[i] == 'linear':
				cube[i] *= pmaxs[i] - pmins[i]
				cube[i] += pmins[i]
			elif types[i] == 'log':
				lmin = np.log(np.abs(pmins[i]))
				lmax = np.log(np.abs(pmaxs[i]))			
				cube[i] *= lmax - lmin
				cube[i] += lmin
				cube[i] = np.exp(cube[i])
			elif types[i] == 'fixed':
				cube[i] = pmins[i]
		return cube
	return prior

# Define log-likelihood function
def loglike_call(data):
	def loglike(cube, ndim, nparams):

		# Define data
		x = data[0]
		y = data[1]
		e_x = data[2]
		e_y = data[3]

		# Define parameters
		mu_x = cube[0]
		mu_y = cube[1]
		scat_x = cube[2]
		scat_y = cube[3]
		if nparams > 4:
			rho_xy = cube[4]
		else:
			rho_xy = 0.

		# Define covariance and precision matrix
		x_mat = np.diag(np.power(e_x,2)+np.power(scat_x,2))
		y_mat = np.diag(np.power(e_y,2)+np.power(scat_y,2))
		xy_mat = np.diag([rho_xy*scat_x*scat_y]*np.ones(len(x)))
		tmp_1 = np.concatenate((x_mat,xy_mat),axis=0)
		tmp_2 = np.concatenate((xy_mat,y_mat),axis=0)
		cov_mat = np.concatenate((tmp_1,tmp_2),axis=1)
		prec_mat = linalg.inv(cov_mat)
		eigen = np.real(linalg.eig(cov_mat)[0])
		logdetcov = np.sum(np.log(eigen[(eigen>0)]))

		# Calculate log likelihood
		tmp_1 = np.concatenate((x,y)).view(np.matrix)
		tmp_2 = np.concatenate((mu_x*np.ones(len(x)),mu_y*np.ones(len(y)))).view(np.matrix)
		chisq = (tmp_1-tmp_2)*prec_mat*np.transpose(tmp_1-tmp_2)
		loglhood = -0.5*chisq
		loglhood -= 0.5*logdetcov

		return loglhood
	return loglike

# Initialize multinest arguments
def initialize_mnest():
    mnest_args = {'LogLikelihood':'',
                  'Prior':'',
                  'n_dims':0,
                  'n_params':0,
                  'n_clustering_params':0,
                  'wrapped_params':None,
                  'importance_nested_sampling':False,
                  'multimodal':False,
                  'const_efficiency_mode':False,
                  'n_live_points':100,
                  'evidence_tolerance':0.5,
                  'sampling_efficiency':'model',
                  'n_iter_before_update':500,
                  'null_log_evidence':-1.e90,
                  'max_modes':100,
                  'mode_tolerance':-1.e90,
                  'outputfiles_basename':'',
                  'seed':-1,
                  'verbose':True,
                  'resume':True,
                  'context':0,
                  'write_output':True,
                  'log_zero':-1.e100,
                  'max_iter':0,
                  'init_MPI':False,
                  'dump_callback':None}
    return mnest_args

def main():

	# Set correlation coefficient
	corr = -0.5

	# Set measurement error
	err = 15.

	# Set seed for pseudo-random generator
	np.random.seed(0)

	# Initialize data
	xx = np.array([0., 100.])
	yy = np.array([0., 100.])
	means = np.array([xx.mean(), yy.mean()])
	stds = np.array([xx.std()/3., yy.std()/3.])
	covs = [[stds[0]**2, stds[0]*stds[1]*corr], 
        	[stds[0]*stds[1]*corr, stds[1]**2]] 
	data = np.random.multivariate_normal(means, covs, 100).T
	
	# Add normal measurement error
	ex = err*np.ones(data[0].shape)
	ey = err*np.ones(data[1].shape)
	data[0] += np.random.normal(0., ex[0], len(data[0]))
	data[1] += np.random.normal(0., ey[0], len(data[1]))		

	# Callculate standard Pearson's rank correlation coefficient
	# pearsonr = stats.pearsonr(data[0],data[1])

	# Set multinest arguments
	mnest_args = initialize_mnest()

	# Run model without correlation
	types = ['linear','linear','log','log']
	pmins = [-1.e2,-1.e2,1.e-2,1.e-2]
	pmaxs = [1.e2,1.e2,1.e2,1.e2]
	n_params = len(types)
	mnest_args['n_params'] = n_params
	mnest_args['n_dims'] = n_params
	mnest_args['outputfiles_basename'] = 'chains/nocorr_'
	mnest_args['LogLikelihood'] = loglike_call([data[0],data[1],ex,ey])
	mnest_args['Prior'] = prior_call(types, pmins, pmaxs)
	pymultinest.run(**mnest_args)
	nocorr_analysis = pymultinest.Analyzer(n_params = mnest_args['n_params'], outputfiles_basename=mnest_args['outputfiles_basename'])

	# Run model with correlation
	types = ['linear','linear','log','log','linear']
	pmins = [-1.e2,-1.e2,1.e-2,1.e-2,-1.]
	pmaxs = [1.e2,1.e2,1.e2,1.e2,1.]
	n_params = len(types)
	mnest_args['n_params'] = n_params
	mnest_args['n_dims'] = n_params
	mnest_args['outputfiles_basename'] = 'chains/withcorr_'
	mnest_args['LogLikelihood'] = loglike_call([data[0],data[1],ex,ey])
	mnest_args['Prior'] = prior_call(types, pmins, pmaxs)
	pymultinest.run(**mnest_args)
	withcorr_analysis = pymultinest.Analyzer(n_params = mnest_args['n_params'], outputfiles_basename=mnest_args['outputfiles_basename'])

	# Plot data and best fitting ellipse
	plt.ioff()
	fig = plt.figure(figsize=(10,8))
	plt.rc('xtick', labelsize=25)
	plt.rc('ytick', labelsize=25) 

	# Set dimension of figure
	gs = gridspec.GridSpec(1,1)
	gs.update(wspace=0.0, hspace=0.0)

	# Initialize subplot
	ax = plt.subplot(gs[0])

	# Add plot data
	ax.errorbar(data[0],data[1],xerr=ex,yerr=ey,linestyle='none',color='k',marker='.',linewidth=2)

	# Add best fitting ellipse
	# best_fit = withcorr_analysis.get_best_fit()
	# ell_x = best_fit['parameters'][0]
	# ell_y = best_fit['parameters'][1]
	# ell_sigx = best_fit['parameters'][2]
	# ell_sigy = best_fit['parameters'][3]
	# ell_rhoxy = best_fit['parameters'][4]
	# ell_angle = 0.5*np.arctan(2.*ell_rhoxy*ell_sigx*ell_sigy/(ell_sigx**2-ell_sigy**2))
	# ell_scale = 1.
	# ell_dx = 2.*ell_scale*ell_sigx
	# ell_dy = 2.*ell_scale*ell_sigy
	# ell_width = ell_dx/np.cos(ell_angle)
	# ell_height = ell_dy/np.cos(ell_angle)	
	# ellipse = patches.Ellipse(xy=(ell_x, ell_y),width=ell_width,height=ell_height,angle=(ell_angle)/np.pi*180.,facecolor='none',edgecolor='r',linewidth=2)

	# Add ellipse
	# ax.add_artist(ellipse)

	# Add axis parameters
	xmin = -49.9 # min(np.hstack([ell_x-1.5*0.5*ell_dx,data[0]-1.5*ex]))
	xmax = 149.9 # max(np.hstack([ell_x+1.5*0.5*ell_dx,data[0]+1.5*ex]))		
	ymin = -49.9 # min(np.hstack([ell_y-1.5*0.5*ell_dy,data[1]-1.5*ey]))
	ymax = 149.9 # max(np.hstack([ell_y+1.5*0.5*ell_dy,data[1]+1.5*ey]))
	xlim = [xmin, xmax]
	ylim = [ymin, ymax]
	ax.set_xscale('linear')
	ax.set_yscale('linear')
	labh = ax.set_xlabel(r'$x$',fontsize=25)
	labh = ax.set_ylabel(r'$y$',fontsize=25)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.minorticks_on()
	ax.tick_params(bottom=True,left=True,top=True,right=True,length=10,width=1,which='major',direction='in')
	ax.tick_params(bottom=True,left=True,top=True,right=True,length=5,width=1,which='minor',direction='in')
	
	plt.savefig('data.pdf')
	plt.close(fig)

	# Plot correlation probability distribution
	plt.ioff()
	fig = plt.figure(figsize=(10,8))
	plt.rc('xtick', labelsize=25)
	plt.rc('ytick', labelsize=25) 

	# Set dimension of figure
	gs = gridspec.GridSpec(1,1)
	gs.update(wspace=0.0, hspace=0.0)

	# Initialize subplot
	ax = plt.subplot(gs[0])

	# Add plot data
	rho = withcorr_analysis.get_data().T[6]
	weights = withcorr_analysis.get_data().T[0]			
	bins = np.arange(-1.,1.05,0.05)
	low = withcorr_analysis.get_stats()['marginals'][4]['1sigma'][0]
	high = withcorr_analysis.get_stats()['marginals'][4]['1sigma'][1]
	truths = (bins>=low)&(bins<=high)
	hist,edges = np.histogram(rho,bins=bins,weights=weights,normed=True)	
	ax.bar(0.5*(edges[:-1]+edges[1:]),hist,width=np.diff(edges),edgecolor=[0.75,0.75,0.75],facecolor=[0.75,0.75,0.75],linewidth=2,hatch=None,zorder=0)
	# ax.axvline(pearsonr[0],linewidth=2,linestyle='--',color='r',zorder=1)
	ax.axvline(corr,linewidth=2,linestyle='-',color='r',zorder=1)	
	low = withcorr_analysis.get_stats()['marginals'][4]['1sigma'][0]
	high = withcorr_analysis.get_stats()['marginals'][4]['1sigma'][1]
	median = withcorr_analysis.get_stats()['marginals'][4]['median']
	ax.axvline(median,linewidth=2,linestyle='--',color='k',zorder=1)
	ax.axvline(low,linewidth=2,linestyle=':',color='k',zorder=1)
	ax.axvline(high,linewidth=2,linestyle=':',color='k',zorder=1)

	# Add axis parameters
	xlim = [-1., 1.]
	ylim = [0., np.max(hist)*1.2]
	ax.set_xscale('linear')
	ax.set_yscale('linear')
	labh = ax.set_ylabel(r'$p(\rho_{x,y}|\boldsymbol{d_{x}},\boldsymbol{\sigma_{x}},\boldsymbol{d_{y}},\boldsymbol{\sigma_{y}},\mathcal{M})$',fontsize=25)
	labh = ax.set_xlabel(r'$\rho_{x,y}$',fontsize=25)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.minorticks_on()
	ax.tick_params(bottom=True,left=True,top=True,right=True,length=10,width=1,which='major',direction='in')
	ax.tick_params(bottom=True,left=True,top=True,right=True,length=5,width=1,which='minor',direction='in')

	# Add text with probability of correlated model
	logZ = withcorr_analysis.get_stats()['global evidence'] - nocorr_analysis.get_stats()['global evidence']
	prob = np.exp(logZ)/(1.+np.exp(logZ))*100.
	print 'rho_3sig = %.8e - %.8e, Z = %.8e, prob = %.15e' % (low,high,np.exp(logZ),prob)
	ax.text(0.,np.max(hist)*1.1,'$\mathrm{Pr}(\mathcal{M}_{\\rho}) = %.0f\,\mathrm{per\,cent}$'%(prob),fontsize=25)

	plt.savefig('corr_pdf.pdf')
	plt.close(fig)

if __name__ == "__main__": 
    main()
