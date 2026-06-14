import numpy as np
from scipy.special import softmax
import argparse
import logging
import pandas as pd

def title():
    logging.info('**********************************************************************')
    logging.info('* RSparsePro for robust fine-mapping in the presence of LD mismatch  *')
    logging.info('* Version 1.0.0                                                      *')
    logging.info('* (C) Wenmin Zhang (wenmin.zhang@mail.mcgill.ca)                     *')
    logging.info('**********************************************************************')

class RSparsePro(object):						# Variational fine-mapping model with K latent causal-effect components
    def __init__(self, P, K, R, vare):					# Initialize model parameters and variational quantities
        self.p = P							# Number of variants in the locus
        self.k = K							# Maximum number of latent causal effects
        self.vare = vare						# LD mismatch parameter
        if vare != 0:							# Precompute matrix used in z-score denoising/update
            self.mat = np.dot(R, np.linalg.inv(np.eye(self.p) + 1/vare * R))
        self.beta_mu = np.zeros([self.p, self.k])			# Variational posterior means of effect sizes for all SNP-effect pairs
        self.gamma = np.zeros([self.p, self.k])				# Variational posterior assignment probabilities for all SNP-effect pairs
        self.tilde_b = np.zeros((self.p,))				# Latent denoised z-score vector

    def infer_q_beta(self, R):						# Coordinate-ascent update for variational effect-size distributions
        for k in range(self.k):
            idxall = [x for x in range(self.k)]
            idxall.remove(k)
            beta_all_k = (self.gamma[:, idxall] * self.beta_mu[:, idxall]).sum(axis=1)
            res_beta = self.tilde_b - np.dot(R, beta_all_k)		# Residual signal attributable to current effect
            self.beta_mu[:, k] = res_beta				# Update posterior mean effect size for component k
            u = 0.5 * self.beta_mu[:, k] ** 2				# Unnormalized log-posterior score for assigning SNPs to effect k
            self.gamma[:, k] = softmax(u)				# Normalize into posterior assignment probabilities across SNPs

    def infer_tilde_b(self, bhat):					# Update latent adjusted z-scores accounting for LD mismatch
        if self.vare == 0:						# Standard SparsePro setting without LD mismatch correction
            self.tilde_b = bhat
        else:								# RSparsePro setting
            beta_all = (self.gamma * self.beta_mu).sum(axis=1)		# Current expected total genetic effect
            self.tilde_b = np.dot(self.mat, (1/self.vare * bhat + beta_all))	# Posterior update of adjusted z-scores

    def train(self, bhat, R, maxite, eps, ubound):			# Main variational inference loop
        for ite in range(maxite):
            old_gamma = self.gamma.copy()
            old_beta = self.beta_mu.copy()
            old_tilde = self.tilde_b.copy()
            self.infer_tilde_b(bhat)					# Update adjusted z-scores
            self.infer_q_beta(R)					# Update variational effect distributions
            diff_gamma = np.linalg.norm(self.gamma-old_gamma)		
            diff_beta = np.linalg.norm(self.beta_mu - old_beta)
            diff_b = np.linalg.norm(self.tilde_b - old_tilde)
            all_diff = diff_gamma + diff_beta + diff_b			# Aggregate convergence criterion
            logging.info('Iteration-->{} . Diff_b: {:.1f} . Diff_s: {:.1f} . Diff_mu: {:.1f} . ALL: {:.1f}'.format(ite, diff_b, diff_gamma, diff_beta, all_diff))
            if all_diff < eps:						# Check convergence threshold
                logging.info("The RSparsePro algorithm has converged.")
                converged = True
                break
            if ite == (maxite - 1) or abs(all_diff) > ubound:
                logging.info("The RSparsePro algorithm didn't converge.")
                converged = False
                break
        return converged

    def get_PIP(self):							# Compute SNP-level posterior inclusion probabilities
        return np.max((self.gamma), axis=1).round(4)

    def get_effect(self, cthres):					# Construct credible sets from variational assignments
        vidx = np.argsort(-self.gamma, axis=1)				# For each SNP, rank latent effects by posterior probability
        matidx = np.argsort(-self.gamma, axis=0)			# For each effect, rank SNPs by posterior probability
        mat_eff = np.zeros((self.p, self.k))
        for p in range(self.p):						# Keep only strongest effect assignment per SNP
            mat_eff[p, vidx[p, 0]] = self.gamma[p, vidx[p, 0]]
        mat_eff[mat_eff < 1/(self.p+1)] = 0
        csum = mat_eff.sum(axis=0).round(2)				# Total attainable coverage for each effect group
        logging.info("Attainable coverage for effect groups: {}".format(csum))
        eff = {}
        eff_gamma = {}
        eff_mu = {}
        for k in range(self.k):						# Evaluate each latent effect component
            if csum[k] >= cthres:
                p = 0
                while np.sum(mat_eff[matidx[0:p, k], k]) < cthres * csum[k]:	# Expand set until desired coverage achieved
                    p = p + 1
                cidx = matidx[0:p, k].tolist()
                eff[k] = cidx
                eff_gamma[k] = mat_eff[cidx, k].round(4)
                eff_mu[k] = self.beta_mu[cidx, k].round(4)
        return eff, eff_gamma, eff_mu					# Return credible set summaries

    def get_ztilde(self):
        return self.tilde_b.round(4)

def get_eff_maxld(eff, ld):						# Compute the maximum LD between lead variants from different effect groups
    idx = [i[0] for i in eff.values()]
    if len(eff)>1:
        maxld = np.abs(np.tril(ld[np.ix_(idx,idx)],-1)).max()
    else:
        maxld = 0.0
    return maxld

def get_eff_minld(eff, ld):						# Compute the minimum LD within each effect group
    if len(eff)==0:
        minld = 1.0
    else:
        minld = min([abs(ld[np.ix_(v, v)]).min() for _,v in eff.items()])
    return minld

def get_ordered(eff_mu):
    if len(eff_mu)>1:
        val_mu = [round(-abs(i[0])) for _,i in eff_mu.items()]
        ordered = (list(eff_mu.keys())[-1] == len(eff_mu)-1) #and (sorted(val_mu) == val_mu)
    else:
        ordered = True
    return ordered

def adaptive_train(zscore, ld, K, maxite, eps, ubound, cthres, minldthres, maxldthres, eincre, varemax, varemin):	# RSparsePro fitting with automatic LD-mismatch tuning
    vare = 0
    mc = False
    while (not mc) or (not get_ordered(eff_mu)) or (minld < minldthres) or (maxld > maxldthres):	# Continue until convergence
        model = RSparsePro(len(zscore), K, ld, vare)			# Instantiate variational model using current mismatch parameter
        mc = model.train(zscore, ld, maxite, eps, ubound)		# Run variational inference
        eff, eff_gamma, eff_mu = model.get_effect(cthres)		# Extract inferred effect groups and credible sets
        maxld = get_eff_maxld(eff, ld)					# Compute the maximum LD between lead variants from different effect groups
        minld = get_eff_minld(eff, ld)					# Compute the minimum LD within each effect group
        logging.info("Max ld across effect groups: {}.".format(maxld))
        logging.info("Min ld within effect groups: {}.".format(minld))
        logging.info("vare = {}".format(round(vare,4)))
        if vare > varemax or (len(eff)<2 and get_ordered(eff_mu)):	# Stopping criteria
            model = RSparsePro(len(zscore), 1, ld, 0)			# Fall back to single-effect SparsePro model
            mc = model.train(zscore, ld, maxite, eps, ubound)
            eff, eff_gamma, eff_mu = model.get_effect(cthres)
            break
        elif vare ==0:
            vare = varemin
        else:
            vare *= eincre
    ztilde = model.get_ztilde()						# Retrieve final adjusted/denoised z-scores
    PIP = model.get_PIP()						# Compute SNP-level posterior inclusion probabilities
    return eff, eff_gamma, eff_mu, PIP, ztilde

def parse_args():
    parser = argparse.ArgumentParser(description='RSparsePro Commands:')
    parser.add_argument('--z', type=str, default=None, help='path to summary statistics', required=True)
    parser.add_argument('--ld', type=str, default=None, help='path to ld matrix', required=True)
    parser.add_argument('--save', type=str, default=None, help='path to save results', required=True)
    parser.add_argument('--K', type=int, default=10, help='largest number of causal signals')
    parser.add_argument('--maxite', type=int, default=100, help='max number of iterations')
    parser.add_argument('--eps', type=float, default=1e-5, help='convergence criterion')
    parser.add_argument('--ubound', type=int, default=100000, help='upper bound for convergence')
    parser.add_argument('--cthres', type=float, default=0.95, help='attainable coverage threshold for effect groups')
    parser.add_argument('--eincre', type=float, default=1.5, help='adjustment for error parameter')
    parser.add_argument('--minldthres', type=float, default=0.7, help='ld within effect groups')
    parser.add_argument('--maxldthres', type=float, default=0.2, help='ld across effect groups')
    parser.add_argument('--varemax', type=float, default=100.0, help='max error parameter')
    parser.add_argument('--varemin', type=float, default=1e-3, help='min error parameter')
    args = parser.parse_args()
    return args

def print_args(args):
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(filename='{}.rsparsepro.log'.format(args.save), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # 
    title()
    print_args(args)
    zfile = pd.read_csv(args.z, sep='\t')
    ld = pd.read_csv(args.ld, sep='\s+', header=None).fillna(0).values
    eff, eff_gamma, eff_mu, PIP, ztilde = adaptive_train(zfile['Z'], ld, args.K, args.maxite, args.eps, args.ubound, args.cthres, args.minldthres, args.maxldthres, args.eincre, args.varemax, args.varemin)
    zfile['PIP'] = PIP
    zfile['z_estimated'] = ztilde
    zfile['cs'] = 0
    for e in eff:
        mcs_idx = [zfile['RSID'][j] for j in eff[e]]
        logging.info(f'The {e}-th effect group contains effective variants:')
        logging.info(f'causal variants: {mcs_idx}')
        logging.info(f'variant probabilities for this effect group: {eff_gamma[e]}')
        logging.info(f'zscore for this effect group: {eff_mu[e]}\n')
        zfile.iloc[eff[e], zfile.columns.get_loc('cs')] = e+1
    zfile.to_csv('{}.rsparsepro.txt'.format(args.save), sep='\t', header=True, index=False)
