""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
from networkx.algorithms.components.connected import connected_components

import sklearn.metrics as metrics
from functools import partial
from tensorboardX import SummaryWriter

import sys
import time
import math
import pickle
import shutil
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
import causaleffect
from torch import nn, optim
from gae.model import VGAE3MLP
from gae.optimizer import loss_function as gae_loss

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain


decimal_round = lambda x: round(x, 5)
color_map = ['gray', 'blue', 'purple', 'red', 'brown', 'green', 'orange', 'olive']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity', help='Name of dataset.')
parser.add_argument('--output', type=str, default=None, help='output path.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='Number of samples in a minibatch.')
parser.add_argument('--seed', type=int, default=42, help='Number of training epochs.')
parser.add_argument('--max_grad_norm', type=float, default=1, help='max_grad_norm.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--encoder_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--encoder_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--encoder_output', type=int, default=16, help='Dim of output of VGAE encoder.')
parser.add_argument('--decoder_hidden1', type=int, default=16, help='Number of units in decoder hidden layer 1.')
parser.add_argument('--decoder_hidden2', type=int, default=16, help='Number of units in decoder  hidden layer 2.')
parser.add_argument('--K', type=int, default=8, help='Number of casual factors.')
parser.add_argument('--coef_lambda', type=float, default=0.01, help='Coefficient of gae loss.')
parser.add_argument('--coef_kl', type=float, default=0.01, help='Coefficient of gae loss.')
parser.add_argument('--coef_causal', type=float, default=1.0, help='Coefficient of causal loss.')
parser.add_argument('--coef_size', type=float, default=0.0, help='Coefficient of size loss.')
parser.add_argument('--NX', type=int, default=1, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--NA', type=int, default=1, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--Nalpha', type=int, default=25, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--Nbeta', type=int, default=100, help='Number of monte-carlo samples per noncausal factor.')
parser.add_argument('--node_perm', action="store_true", help='Use node permutation as data augmentation for causal training.')
parser.add_argument('--load_ckpt', default=None, help='Load parameters from checkpoint.')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--patient', type=int, default=100, help='Patient for early stopping.')
parser.add_argument('--plot_info_flow', action='store_true')

args = parser.parse_args()

if args.gpu and torch.cuda.is_available():
    print("Use cuda")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

def graph_labeling(G):
    for node in G:
        G.nodes[node]['string'] = 1
    old_strings = tuple([G.nodes[node]['string'] for node in G])
    for iter_num in range(100):
        for node in G:
            string = sorted([G.nodes[neigh]['string'] for neigh in G.neighbors(node)])
            G.nodes[node]['concat_string'] =  tuple([G.nodes[node]['string']] + string)
        d = nx.get_node_attributes(G,'concat_string')
        nodes,strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        map_string = dict([[string, i+1] for i, string in enumerate(sorted(set(strings)))])
        for node in nodes:
            G.nodes[node]['string'] = map_string[G.nodes[node]['concat_string']]
        new_strings = tuple([G.nodes[node]['string'] for node in G])
        if old_strings == new_strings:
            break
        else:
            old_strings = new_strings
    return G

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.from_numpy(adj_normalized).float()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def gaeloss(x,mu,logvar,data):
    return gae_loss(preds=x, labels=data['adj_label'],
                    mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                    norm=data['norm'], pos_weight=data['pos_weight'])

softmax = torch.nn.Softmax(dim=1)
ce = torch.nn.CrossEntropyLoss(reduction='mean')

def main():
    # Load a model checkpoint
    ckpt = torch.load('ckpt/%s_base_h20_o20.pth.tar'%(args.dataset))
    cg_dict = ckpt["cg"] # get computation graph
    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Explain Graph prediction
    classifier = models.GcnEncoderGraph(
        input_dim=input_dim,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        args=argparse.Namespace(gpu=args.gpu,bias=True,method=None),
    ).to(device)

    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()
    print("Number of graphs:", cg_dict["adj"].shape[0])
    if args.output is None:
        args.output = args.dataset

    K = args.K
    L = args.encoder_output - K
    ceparams = {
        'Nalpha': args.Nalpha,
        'Nbeta' : args.Nbeta,
        'K'     : K,
        'L'     : L,
        'z_dim' : args.encoder_output,
        'M'     : num_classes}

    model = VGAE3MLP(
        input_dim + 100, args.encoder_hidden1, args.encoder_hidden1,
        args.encoder_output, args.decoder_hidden1, args.decoder_hidden2,
        args.K, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = gaeloss
    label_onehot = torch.eye(100, dtype=torch.float)
    class GraphSampler(torch.utils.data.Dataset):
        """ Sample graphs and nodes in graph
        """
        def __init__(
            self,
            graph_idxs
        ):
            self.graph_idxs = graph_idxs
            self.graph_data = []
            for graph_idx in graph_idxs:
                adj = cg_dict["adj"][graph_idx].float()
                label = cg_dict["label"][graph_idx].long()
                feat = cg_dict["feat"][graph_idx, :].float()
                G = graph_labeling(nx.from_numpy_array(cg_dict["adj"][graph_idx].numpy()))
                graph_label = np.array([G.nodes[node]['string'] for node in G])
                graph_label_onehot = label_onehot[graph_label]
                sub_feat = torch.cat((feat, graph_label_onehot), dim=1)
                adj_label = adj + np.eye(adj.shape[0])
                n_nodes = adj.shape[0]
                graph_size = torch.count_nonzero(adj.sum(-1))
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                pos_weight = torch.from_numpy(np.array(pos_weight))
                norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
                self.graph_data += [{
                    "graph_idx": graph_idx,
                    "graph_size": graph_size, 
                    "sub_adj": adj.to(device), 
                    "feat": feat.to(device).float(), 
                    "sub_feat": sub_feat.to(device).float(), 
                    "sub_label": label.to(device).float(), 
                    "adj_label": adj_label.to(device).float(),
                    "n_nodes": torch.Tensor([n_nodes])[0].to(device),
                    "pos_weight": pos_weight.to(device),
                    "norm": norm.to(device)
                }]

        def __len__(self):
            return len(self.graph_idxs)

        def __getitem__(self, idx):
            return self.graph_data[idx]

    train_idxs = np.array(cg_dict['train_idx'])
    val_idxs = np.array(cg_dict['val_idx'])
    test_idxs = np.array(cg_dict['test_idx'])
    train_graphs = GraphSampler(train_idxs)
    train_dataset = torch.utils.data.DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_graphs = GraphSampler(val_idxs)
    val_dataset = torch.utils.data.DataLoader(
        val_graphs,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )
    test_graphs = GraphSampler(test_idxs)
    test_dataset = torch.utils.data.DataLoader(
        test_graphs,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    def eval_model(dataset, prefix=''):
        model.eval()
        with torch.no_grad():
            for data in dataset:
                labels = cg_dict['label'][data['graph_idx'].long()].long().to(device)
                recovered, mu, logvar = model(data['sub_feat'], data['sub_adj'])
                recovered_adj = torch.sigmoid(recovered)
                nll_loss =  criterion(recovered, mu, logvar, data).mean()
                org_adjs = data['sub_adj']
                org_logits = classifier(data['feat'], data['sub_adj'])[0]
                org_probs = F.softmax(org_logits, dim=1)
                org_log_probs = F.log_softmax(org_logits, dim=1)
                masked_recovered_adj = recovered_adj * data['sub_adj']
                recovered_logits = classifier(data['feat'], masked_recovered_adj)[0]
                recovered_probs = F.softmax(recovered_logits, dim=1)
                recovered_log_probs = F.log_softmax(recovered_logits, dim=1)
                alpha_mu = torch.zeros_like(mu)
                alpha_mu[:,:,:args.K] = mu[:,:,:args.K]
                alpha_adj = torch.sigmoid(model.dc(alpha_mu))
                masked_alpha_adj = alpha_adj * data['sub_adj']
                alpha_logits = classifier(data['feat'], masked_alpha_adj)[0]
                beta_mu = torch.zeros_like(mu)
                beta_mu[:,:,args.K:] = mu[:,:,args.K:]
                beta_adj = torch.sigmoid(model.dc(beta_mu))
                masked_beta_adj = beta_adj * data['sub_adj']
                beta_logits = classifier(data['feat'], masked_beta_adj)[0]
                causal_loss = []
                beta_info = []
                
                for idx in random.sample(range(0, data['feat'].shape[0]), args.NX):                 
                    _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, data['sub_adj'][idx], data['feat'][idx], act=torch.sigmoid, device=device)
                    _beta_info, _ = causaleffect.beta_info_flow(ceparams, model.dc, classifier, data['sub_adj'][idx], data['feat'][idx], act=torch.sigmoid, device=device)
                    causal_loss += [_causal_loss]
                    beta_info += [_beta_info]
                    for A_idx in random.sample(range(0, data['feat'].shape[0]), args.NA-1):
                        if args.node_perm:
                            perm = torch.randperm(data['graph_size'][idx])
                            perm_adj = data['sub_adj'][idx].clone().detach()
                            perm_adj[:data['graph_size'][idx]] = perm_adj[perm]
                        else:
                            perm_adj = data['sub_adj'][A_idx]
                        _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, perm_adj, data['feat'][idx], act=torch.sigmoid, device=device)
                        _beta_info, _ = causaleffect.beta_info_flow(ceparams, model.dc, classifier, perm_adj, data['feat'][idx], act=torch.sigmoid, device=device)
                        causal_loss += [_causal_loss]
                        beta_info += [_beta_info]
                causal_loss = torch.stack(causal_loss).mean()
                alpha_info = causal_loss
                beta_info = torch.stack(beta_info).mean()
                klloss = F.kl_div(F.log_softmax(alpha_logits, dim=1), org_probs, reduction='mean')
            pred_labels = torch.argmax(org_probs,axis=1)
            org_acc = (torch.argmax(org_probs,axis=1) == torch.argmax(recovered_probs,axis=1)).float().mean()
            pred_acc = (torch.argmax(recovered_probs,axis=1) == labels).float().mean()
            kl_pred_org = F.kl_div(recovered_log_probs, org_probs, reduction='mean')
            alpha_probs = F.softmax(alpha_logits, dim=1)
            alpha_log_probs = F.log_softmax(alpha_logits, dim=1)
            beta_probs = F.softmax(beta_logits, dim=1)
            beta_log_probs = F.log_softmax(beta_logits, dim=1)
            alpha_gt_acc = (torch.argmax(alpha_probs,axis=1) == labels).float().mean()
            alpha_pred_acc = (torch.argmax(alpha_probs,axis=1) == pred_labels).float().mean()
            alpha_kld = F.kl_div(alpha_log_probs, org_probs, reduction='mean')
            beta_gt_acc = (torch.argmax(beta_probs,axis=1) == labels).float().mean()
            beta_pred_acc = (torch.argmax(beta_probs,axis=1) == pred_labels).float().mean()
            beta_kld = F.kl_div(beta_log_probs, org_probs, reduction='mean')
            alpha_sparsity = masked_alpha_adj.mean((1,2))/org_adjs.mean((1,2))
            loss = args.coef_lambda * nll_loss + \
                args.coef_causal * causal_loss + \
                args.coef_kl * klloss + \
                args.coef_size * alpha_sparsity.mean()
            writer.add_scalar("%s/total_loss"%prefix, loss, epoch)
            writer.add_scalar("%s/nll"%prefix, nll_loss, epoch)
            writer.add_scalar("%s/causal"%prefix, causal_loss, epoch)
            writer.add_scalar("%s/alpha_info_flow"%prefix, alpha_info/(alpha_info+beta_info), epoch)
            writer.add_scalar("%s/beta_info_flow"%prefix, beta_info/(alpha_info+beta_info), epoch)
            writer.add_scalar("%s/acc(Y_rec, Y_org)"%prefix, org_acc, epoch)
            writer.add_scalar("%s/acc(Y_rec, labels)"%prefix, pred_acc, epoch)
            writer.add_scalar("%s/kld(Y_rec, Y_org)"%prefix, kl_pred_org, epoch)
            writer.add_scalar("%s/kld(Y_alpha, Y_org)"%prefix, alpha_kld, epoch)
            writer.add_scalar("%s/kld(Y_beta, Y_org)"%prefix, beta_kld, epoch)
            writer.add_scalar("%s/alpha_sparsity"%prefix, alpha_sparsity.mean(), epoch)
            writer.add_scalar("%s/acc(Y_alpha, labels)"%prefix, alpha_gt_acc, epoch)
            writer.add_scalar("%s/acc(Y_beta, labels)"%prefix, beta_gt_acc, epoch)
            writer.add_scalar("%s/acc(Y_alpha, Y_org)"%prefix, alpha_pred_acc, epoch)
            writer.add_scalar("%s/acc(Y_beta, Y_org)"%prefix, beta_pred_acc, epoch)
        return loss.item()

    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch
        }, filename)

    if args.load_ckpt:
        ckpt_path = args.load_ckpt
    else:
        ckpt_path = os.path.join('explanation', args.output, 'model.ckpt')
    if os.path.exists(ckpt_path) and not args.retrain:
        print("Load checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
    else:
        args.retrain = True
        start_epoch = 1
        best_loss = 100
    if args.resume or args.retrain:
        patient = args.patient
        model.train()
        start_time = time.time()
        writer = SummaryWriter(comment=args.output)
        os.makedirs('explanation/%s' % args.output, exist_ok=True)
        for epoch in tqdm(range(start_epoch, args.epoch+1)):
            # print("------- Epoch %2d ------" % epoch)
            model.train()
            train_losses = []
            for batch_idx, data in enumerate(train_dataset):
                optimizer.zero_grad()
                mu, logvar = model.encode(data['sub_feat'], data['sub_adj'])
                sample_mu = model.reparameterize(mu, logvar)
                recovered = model.dc(sample_mu)
                org_logit = classifier(data['feat'], data['sub_adj'])[0]
                org_probs = F.softmax(org_logit, dim=1)
                if args.coef_lambda:
                    nll_loss = args.coef_lambda * criterion(recovered, mu, logvar, data).mean()
                else:
                    nll_loss = 0
                alpha_mu = torch.zeros_like(sample_mu)
                alpha_mu[:,:,:args.K] = sample_mu[:,:,:args.K]
                alpha_adj = torch.sigmoid(model.dc(alpha_mu))
                masked_alpha_adj = alpha_adj * data['sub_adj']
                alpha_logit = classifier(data['feat'], masked_alpha_adj)[0]
                alpha_sparsity = masked_alpha_adj.mean((1,2))/data['sub_adj'].mean((1,2))
                if args.coef_causal:
                    causal_loss = []
                    NX = min(data['feat'].shape[0], args.NX)
                    NA = min(data['feat'].shape[0], args.NA)
                    for idx in random.sample(range(0, data['feat'].shape[0]), NX):
                        _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, data['sub_adj'][idx], data['feat'][idx], act=torch.sigmoid, device=device)
                        causal_loss += [_causal_loss]
                        for A_idx in random.sample(range(0, data['feat'].shape[0]), NA-1):
                            if args.node_perm:
                                perm = torch.randperm(data['graph_size'][idx])
                                perm_adj = data['sub_adj'][idx].clone().detach()
                                perm_adj[:data['graph_size'][idx]] = perm_adj[perm]
                            else:
                                perm_adj = data['sub_adj'][A_idx]
                            _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, perm_adj, data['feat'][idx], act=torch.sigmoid, device=device)
                            causal_loss += [_causal_loss]
                    causal_loss = args.coef_causal * torch.stack(causal_loss).mean()
                else:
                    causal_loss = 0
                if args.coef_kl:
                    klloss = args.coef_kl * F.kl_div(F.log_softmax(alpha_logit,dim=1), org_probs, reduction='mean')
                else:
                    klloss = 0
                if args.coef_size:
                    size_loss = args.coef_size * alpha_sparsity.mean()
                else:
                    size_loss = 0

                loss = nll_loss + causal_loss + klloss + size_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_losses += [[nll_loss, causal_loss, klloss, size_loss]]
                sys.stdout.flush()
            
            # train_loss = (torch.cat(train_losses)).mean().item()
            nll_loss, causal_loss, klloss, size_loss = torch.tensor(train_losses).mean(0)
            writer.add_scalar("train/nll", nll_loss, epoch)
            writer.add_scalar("train/causal", causal_loss, epoch)
            writer.add_scalar("train/kld(Y_alpha,Y_org)", klloss, epoch)
            writer.add_scalar("train/alpha_sparsity", size_loss, epoch)
            writer.add_scalar("train/total_loss", nll_loss + causal_loss + klloss + size_loss, epoch)

            val_loss = eval_model(val_dataset, 'val')
            patient -= 1
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint('explanation/%s/model.ckpt' % args.output)
                test_loss = eval_model(test_dataset, 'test')
                patient = 100
            elif patient <= 0:
                print("Early stopping!")
                break
            if epoch % 100 == 0:
                save_checkpoint('explanation/%s/model-%depoch.ckpt' % (args.output,epoch))
        print("Train time:", time.time() - start_time)
        writer.close()
        checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("Start evaluation.")

    model.eval()
    results = []
    with torch.no_grad():
        for data in test_dataset:
            labels = cg_dict['label'][data['graph_idx'].long()].long().to(device)
            mu, logvar = model.encode(data['sub_feat'], data['sub_adj'])
            org_logits = classifier(data['feat'], data['sub_adj'])[0]
            org_probs = F.softmax(org_logits, dim=1)
            pred_labels = torch.argmax(org_probs,axis=1)
            alpha_mu = torch.zeros_like(mu)
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            alpha_mu[:,:,:args.K] = eps.mul(std).add_(mu)[:,:,:args.K]
            alpha_adj = torch.sigmoid(model.dc(alpha_mu))
            masked_alpha_adj = alpha_adj * data['sub_adj']
            flatten_alpha_adj = masked_alpha_adj.flatten(1)
            for sparsity in np.arange(0, 1, 0.05):
                topk = torch.round(data['sub_adj'].sum((1,2)) * sparsity).long().unsqueeze(-1)
                threshold = torch.gather(flatten_alpha_adj.sort(1,descending=True).values, 1, topk)
                threshold = torch.maximum(threshold, torch.ones_like(threshold)*1E-6)
                topk_alpha_adj = (flatten_alpha_adj > threshold).float().view(data['sub_adj'].shape)
                alpha_logits = classifier(data['feat'], topk_alpha_adj)[0]
                alpha_log_probs = F.log_softmax(alpha_logits, dim=1)
                results += [{
                    "sparsity": sparsity,
                    "alpha_topk": topk_alpha_adj.sum((1,2)).mean().item()/2,
                    "alpha_sparsity": (topk_alpha_adj.sum((1,2))/data['sub_adj'].sum((1,2))).mean().item(),
                    "alpha_gt_acc": (torch.argmax(alpha_logits,axis=1) == labels).float().mean().item(),
                    "alpha_pred_acc": (torch.argmax(alpha_logits,axis=1) == pred_labels).float().mean().item(),
                    "alpha_kld": F.kl_div(alpha_log_probs, org_probs, reduction='batchmean').item()
                }]
    columns = results[0].keys()
    df = pd.DataFrame(results, columns = columns)
    df.to_csv(os.path.join('explanation', args.output, 'results.csv'))
    print(df)
    
    if args.plot_info_flow:
        print("Calculating information flow...")
        with torch.no_grad():
            infos = [
                [
                    - causaleffect.joint_uncond_singledim(
                        ceparams, model.dc, classifier, 
                        data['sub_adj'][idx], data['feat'][idx], 
                        dim, act=torch.sigmoid, device=device
                    )[0] for dim in range(ceparams['z_dim'])
                ] for idx in tqdm(range(data['feat'].shape[0]))
            ]
            infos = torch.tensor(infos, device=device)
            infos = F.normalize(infos, p=1, dim=1)
            print(infos.mean(0))
        results = []
        for info in infos:
            for dim in range(ceparams['z_dim']):
                results += [{'dim': dim+1, 'info': info[dim].item()}]
        df = pd.DataFrame(results, columns = results[0].keys())
        df.to_csv(os.path.join('explanation', args.output, 'info_flow.csv'))
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        colors = ["red", "blue", "orange", "green"]
        customPalette = sns.set_palette(sns.color_palette(colors))
        matplotlib.rcParams.update({'font.size': 16})
        plt.rcParams["font.family"] = "Times New Roman"
        f = plt.figure(figsize=(7,5))
        ax = sns.barplot(data=df, x='dim', y='info', palette=customPalette)
        plt.xlabel('Z [i]')
        plt.ylabel('Information Measurements')
        f.savefig(os.path.join('explanation', args.output, 'info_flow.pdf'))
        plt.show()

if __name__ == "__main__":
    main()

