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

class RSparsePro(object):
    def __init__(self, P, K, R, vare):
        self.p = P
        self.k = K
        self.vare = vare
        if vare != 0:
            self.mat = np.dot(R, np.linalg.inv(np.eye(self.p) + 1/vare * R))
        self.beta_mu = np.zeros([self.p, self.k])
        self.gamma = np.zeros([self.p, self.k])
        self.tilde_b = np.zeros((self.p,))

    def infer_q_beta(self, R):
        for k in range(self.k):
            idxall = [x for x in range(self.k)]
            idxall.remove(k)
            beta_all_k = (self.gamma[:, idxall] * self.beta_mu[:, idxall]).sum(axis=1)
            res_beta = self.tilde_b - np.dot(R, beta_all_k)
            self.beta_mu[:, k] = res_beta
            u = 0.5 * self.beta_mu[:, k] ** 2
            self.gamma[:, k] = softmax(u)

    def infer_tilde_b(self, bhat):
        if self.vare == 0:
            self.tilde_b = bhat
        else:
            beta_all = (self.gamma * self.beta_mu).sum(axis=1)
            self.tilde_b = np.dot(self.mat, (1/self.vare * bhat + beta_all))

    def train(self, bhat, R, maxite, eps, ubound):
        for ite in range(maxite):
            old_gamma = self.gamma.copy()
            old_beta = self.beta_mu.copy()
            old_tilde = self.tilde_b.copy()
            self.infer_tilde_b(bhat)
            self.infer_q_beta(R)
            diff_gamma = np.linalg.norm(self.gamma-old_gamma)
            diff_beta = np.linalg.norm(self.beta_mu - old_beta)
            diff_b = np.linalg.norm(self.tilde_b - old_tilde)
            all_diff = diff_gamma + diff_beta + diff_b
            logging.info('Iteration-->{} . Diff_b: {:.1f} . Diff_s: {:.1f} . Diff_mu: {:.1f} . ALL: {:.1f}'.format(ite, diff_b, diff_gamma, diff_beta, all_diff))
            if all_diff < eps:
                logging.info("The RSparsePro algorithm has converged.")
                converged = True
                break
            if ite == (maxite - 1) or abs(all_diff) > ubound:
                logging.info("The RSparsePro algorithm didn't converge.")
                converged = False
                break
        return converged

    def get_PIP(self):
        return np.max((self.gamma), axis=1).round(4)

    def get_effect(self, cthres):
        vidx = np.argsort(-self.gamma, axis=1)
        matidx = np.argsort(-self.gamma, axis=0)
        mat_eff = np.zeros((self.p, self.k))
        for p in range(self.p):
            mat_eff[p, vidx[p, 0]] = self.gamma[p, vidx[p, 0]]
        mat_eff[mat_eff < 1/(self.p+1)] = 0
        csum = mat_eff.sum(axis=0).round(2)
        logging.info("Attainable coverage for effect groups: {}".format(csum))
        eff = {}
        eff_gamma = {}
        eff_mu = {}
        for k in range(self.k):
            if csum[k] >= cthres:
                p = 0
                while np.sum(mat_eff[matidx[0:p, k], k]) < cthres * csum[k]:
                    p = p + 1
                cidx = matidx[0:p, k].tolist()
                eff[k] = cidx
                eff_gamma[k] = mat_eff[cidx, k].round(4)
                eff_mu[k] = self.beta_mu[cidx, k].round(4)
        return eff, eff_gamma, eff_mu

    def get_ztilde(self):
        return self.tilde_b.round(4)

    #def get_resz(self, bhat, ld, eff):
    #    idx = [i[0] for i in eff.values()]
    #    realmu = np.zeros(len(bhat))
    #    realmu[idx] = np.dot(np.linalg.inv(ld[np.ix_(idx, idx)]), bhat[idx])
    #    estz = np.dot(ld, realmu)
    #    resz = bhat - estz
    #    return resz.round(4)

def get_eff_maxld(eff, ld):
    idx = [i[0] for i in eff.values()]
    if len(eff)>1:
        maxld = np.abs(np.tril(ld[np.ix_(idx,idx)],-1)).max()
    else:
        maxld = 0.0
    return maxld

def get_eff_minld(eff, ld):
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

def adaptive_train(zscore, ld, K, maxite, eps, ubound, cthres, minldthres, maxldthres, eincre, varemax, varemin):
    vare = 0
    mc = False
    while (not mc) or (not get_ordered(eff_mu)) or (minld < minldthres) or (maxld > maxldthres):
        model = RSparsePro(len(zscore), K, ld, vare)
        mc = model.train(zscore, ld, maxite, eps, ubound)
        eff, eff_gamma, eff_mu = model.get_effect(cthres)
        maxld = get_eff_maxld(eff, ld)
        minld = get_eff_minld(eff, ld)
        logging.info("Max ld across effect groups: {}.".format(maxld))
        logging.info("Min ld within effect groups: {}.".format(minld))
        logging.info("vare = {}".format(round(vare,4)))
        if vare > varemax or (len(eff)<2 and get_ordered(eff_mu)):
            #logging.info("Algorithm didn't converge at the max vare. Setting K to 1.")
            model = RSparsePro(len(zscore), 1, ld, 0)
            mc = model.train(zscore, ld, maxite, eps, ubound)
            eff, eff_gamma, eff_mu = model.get_effect(cthres)
            break
        elif vare ==0:
            vare = varemin
        else:
            vare *= eincre
    ztilde = model.get_ztilde()
    #resz = model.get_resz(zscore, ld, eff)
    PIP = model.get_PIP()
    return eff, eff_gamma, eff_mu, PIP, ztilde #resz

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
    #zfile['resz'] = resz
    zfile['cs'] = 0
    for e in eff:
        mcs_idx = [zfile['RSID'][j] for j in eff[e]]
        logging.info(f'The {e}-th effect group contains effective variants:')
        logging.info(f'causal variants: {mcs_idx}')
        logging.info(f'variant probabilities for this effect group: {eff_gamma[e]}')
        logging.info(f'zscore for this effect group: {eff_mu[e]}\n')
        zfile.iloc[eff[e], zfile.columns.get_loc('cs')] = e+1
    zfile.to_csv('{}.rsparsepro.txt'.format(args.save), sep='\t', header=True, index=False)
