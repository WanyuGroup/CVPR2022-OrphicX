import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import shutil
import random
import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp
from tensorboardX import SummaryWriter

import causaleffect
from gae.model import VGAE3MLP
from gae.optimizer import loss_function as gae_loss

sys.path.append('gnnexp')
import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--encoder_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--encoder_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--encoder_output', type=int, default=16, help='Dim of output of VGAE encoder.')
parser.add_argument('--decoder_hidden1', type=int, default=16, help='Number of units in decoder hidden layer 1.')
parser.add_argument('--decoder_hidden2', type=int, default=16, help='Number of units in decoder  hidden layer 2.')
parser.add_argument('--n_hops', type=int, default=3, help='Number of hops.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples in a minibatch.')
parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='syn1', help='Type of dataset.')
parser.add_argument('--output', type=str, default=None, help='Path of output dir.')
parser.add_argument('--load_ckpt', default=None, help='Load parameters from checkpoint.')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--graph_labelling', action='store_true')
parser.add_argument('--K', type=int, default=3, help='Number of alpha dims in z.')
parser.add_argument('--NX', type=int, default=2, help='Number of samples of X.')
parser.add_argument('--Nalpha', type=int, default=25, help='Number of samples of alpha.')
parser.add_argument('--Nbeta', type=int, default=100, help='Number of samples of beta.')
parser.add_argument('--coef_lambda', type=float, default=0.1, help='Coefficient of gae loss.')
parser.add_argument('--coef_kl', type=float, default=0.2, help='Coefficient of gae loss.')
parser.add_argument('--coef_causal', type=float, default=1.0, help='Coefficient of causal loss.')
parser.add_argument('--coef_size', type=float, default=0.1, help='Coefficient of size loss.')
parser.add_argument('--plot_info_flow', action='store_true')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--patient', type=int, default=100, help='Patient for early stopping.')

args = parser.parse_args()
if args.output is None:
    args.output = args.dataset

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
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

def gaeloss(x,mu,logvar,data):
    return gae_loss(preds=x, labels=data['adj_label'],
                    mu=mu, logvar=logvar, n_nodes=data['graph_size'],
                    norm=data['norm'], pos_weight=data['pos_weight'])

def main():
    ckpt = torch.load('ckpt/%s_base_h20_o20.pth.tar'%(args.dataset))
    cg_dict = ckpt["cg"] # get computation graph
    input_dim = cg_dict["feat"].shape[2]
    adj = cg_dict["adj"][0]
    label = cg_dict["label"][0]
    tg_G = torch_geometric.utils.from_networkx(nx.from_numpy_matrix(adj))
    features = torch.tensor(cg_dict["feat"][0], dtype=torch.float)
    num_classes = max(label)+1
    
    input_dim = cg_dict["feat"].shape[2]
    classifier = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        args=argparse.Namespace(gpu=args.gpu,bias=True,method=None)
    ).to(device)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()

    ceparams = {
        'Nalpha': args.Nalpha,
        'Nbeta' : args.Nbeta,
        'K'     : args.K,
        'L'     : args.encoder_output - args.K,
        'z_dim' : args.encoder_output,
        'M'     : num_classes}

    label_onehot = torch.eye(400, dtype=torch.float)
    def extract_neighborhood(node_idx):
        """Returns the neighborhood of a given ndoe."""
        mapping, edge_idxs, node_idx_new, edge_mask = torch_geometric.utils.k_hop_subgraph(int(node_idx), args.n_hops, tg_G.edge_index, relabel_nodes=True)
        node_idx_new = node_idx_new.item()
        sub_adj = torch_geometric.utils.to_dense_adj(edge_idxs)[0]
        adj_norm = preprocess_graph(sub_adj)
        adj_label = sub_adj + np.eye(sub_adj.shape[0])
        pos_weight = float(sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) / sub_adj.sum()
        pos_weight = torch.from_numpy(np.array(pos_weight))
        norm = torch.tensor(sub_adj.shape[0] * sub_adj.shape[0] / float((sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) * 2))
        # Calculate hop_feat:
        pow_adj = ((sub_adj @ sub_adj >=1).float() - np.eye(sub_adj.shape[0]) - sub_adj >=1).float()
        feat = features[mapping]
        sub_feat = feat
        one_hot = torch.zeros((sub_adj.shape[0], ), dtype=torch.float)
        one_hot[node_idx_new] = 1
        hop_feat = [one_hot, sub_adj[node_idx_new], pow_adj[node_idx_new]]
        if args.n_hops == 3:
            pow3_adj = ((pow_adj @ pow_adj >=1).float() - np.eye(pow_adj.shape[0]) - pow_adj >=1).float()
            hop_feat += [pow3_adj[node_idx_new]]
            hop_feat = torch.stack(hop_feat).t()
            sub_feat = torch.cat((sub_feat, hop_feat), dim=1)
        if args.graph_labelling:
            G = graph_labeling(nx.from_numpy_array(sub_adj.numpy()))
            graph_label = np.array([G.nodes[node]['string'] for node in G])
            graph_label_onehot = label_onehot[graph_label]
            sub_feat = torch.cat((sub_feat, graph_label_onehot), dim=1)
        sub_label = torch.from_numpy(label[mapping])
        return {
            "node_idx_new": node_idx_new,
            "feat": feat.unsqueeze(0).to(device),
            "sub_adj": sub_adj.unsqueeze(0).to(device),
            "sub_feat": sub_feat.unsqueeze(0).to(device),
            "adj_norm": adj_norm.unsqueeze(0).to(device),
            "sub_label": sub_label.to(device),
            "mapping": mapping.to(device),
            "adj_label": adj_label.unsqueeze(0).to(device),
            "graph_size": mapping.shape[-1],
            "pos_weight": pos_weight.unsqueeze(0).to(device),
            "norm": norm.unsqueeze(0).to(device)
        }

    def eval_task(node_idx):
        data = dataset[node_idx]
        recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
        recovered_adj = torch.sigmoid(recovered)
        nll_loss = criterion(recovered, mu, logvar, data).mean()
        org_logits = classifier(data['feat'], data['sub_adj'])[0][0, data['node_idx_new']]
        masked_recovered_adj = recovered_adj * data['sub_adj']
        recovered_logits = classifier(data['feat'], masked_recovered_adj)[0][0,data['node_idx_new']]
        alpha_mu = torch.zeros_like(mu)
        alpha_mu[:,:,:args.K] = mu[:,:,:args.K]
        alpha_adj = torch.sigmoid(model.dc(alpha_mu))
        alpha_size = (alpha_adj*data['sub_adj']).sum()
        org_size = data['sub_adj'].sum()
        alpha_sparsity = alpha_size/org_size
        masked_alpha_adj = alpha_adj * data['sub_adj']
        alpha_logits = classifier(data['feat'], masked_alpha_adj)[0][0,data['node_idx_new']]
        beta_mu = torch.zeros_like(mu)
        beta_mu[:,:,args.K:] = mu[:,:,args.K:]
        beta_adj = torch.sigmoid(model.dc(beta_mu))
        masked_beta_adj = beta_adj * data['sub_adj']
        beta_logits = classifier(data['feat'], masked_beta_adj)[0][0,data['node_idx_new']]
        return nll_loss, org_logits, recovered_logits, alpha_logits, beta_logits, alpha_sparsity

    def eval_model(node_idxs, prefix=''):
        with torch.no_grad():
            labels = torch.from_numpy(label[node_idxs]).long().to(device)
            nll_loss, org_logits, recovered_logits, alpha_logits, beta_logits, alpha_sparsity = zip(*map(eval_task, node_idxs))
            causal_loss = []
            beta_info = []
            for idx in random.sample(node_idxs, args.NX):
                _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, dataset[idx]['sub_adj'], dataset[idx]['feat'], node_idx=dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                _beta_info, _ = causaleffect.beta_info_flow(ceparams, model.dc, classifier, dataset[idx]['sub_adj'], dataset[idx]['feat'], node_idx=dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                causal_loss += [_causal_loss]
                beta_info += [_beta_info]
            nll_loss = torch.stack(nll_loss).mean()
            causal_loss = torch.stack(causal_loss).mean()
            alpha_info = causal_loss
            beta_info = torch.stack(beta_info).mean()
            alpha_logits = torch.stack(alpha_logits)
            beta_logits = torch.stack(beta_logits)
            recovered_logits = torch.stack(recovered_logits)
            org_logits = torch.stack(org_logits)
            org_probs = F.softmax(org_logits, dim=1)
            recovered_probs = F.softmax(recovered_logits, dim=1)
            recovered_log_probs = F.log_softmax(recovered_logits, dim=1)
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
            alpha_sparsity = torch.stack(alpha_sparsity).mean()
            loss = args.coef_lambda * nll_loss + \
                args.coef_causal * causal_loss + \
                args.coef_kl * klloss + \
                args.coef_size * alpha_sparsity
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
            writer.add_scalar("%s/alpha_sparsity"%prefix, alpha_sparsity, epoch)
            writer.add_scalar("%s/acc(Y_alpha, labels)"%prefix, alpha_gt_acc, epoch)
            writer.add_scalar("%s/acc(Y_beta, labels)"%prefix, beta_gt_acc, epoch)
            writer.add_scalar("%s/acc(Y_alpha, Y_org)"%prefix, alpha_pred_acc, epoch)
            writer.add_scalar("%s/acc(Y_beta, Y_org)"%prefix, beta_pred_acc, epoch)
        return loss.item()

    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)

    feat_dim = features.shape[-1]
    # hop feature
    feat_dim += args.n_hops + 1
    if args.graph_labelling:
        feat_dim += label_onehot.shape[-1]
    model = VGAE3MLP(
        feat_dim, args.encoder_hidden1, args.encoder_hidden1,
        args.encoder_output, args.decoder_hidden1, args.decoder_hidden2,
        args.K, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_nodes = label.shape[0]
    train_idxs = np.array(cg_dict['train_idx'])
    test_idxs = np.array([i for i in range(num_nodes) if i not in train_idxs])

    # exclude nodes labeled as class 0 and 4
    train_label = label[train_idxs]
    test_label = label[test_idxs]
    if args.dataset == 'syn2':
        train_idxs = train_idxs[np.where(np.logical_and(train_label != 0, train_label != 4))[0]]
        test_idxs = test_idxs[np.where(np.logical_and(test_label != 0, test_label != 4))[0]]
    else:
        train_idxs = train_idxs[np.where(train_label != 0)[0]]
        test_idxs = test_idxs[np.where(test_label != 0)[0]]

    num_train = len(train_idxs)
    num_test = len(test_idxs)
    dataset = dict([[node_idx,extract_neighborhood(node_idx)] for node_idx in train_idxs])
    dataset.update(dict([[node_idx,extract_neighborhood(node_idx)] for node_idx in test_idxs]))
    val_idxs = list(test_idxs[:num_test//2])
    test_idxs = list(test_idxs[num_test//2:])

    criterion = gaeloss

    def train_task(node_idx):
        data = dataset[node_idx]
        mu, logvar = model.encode(data['sub_feat'], data['adj_norm'])
        sample_mu = model.reparameterize(mu, logvar)
        recovered = model.dc(sample_mu)
        nll_loss = criterion(recovered, mu, logvar, data).mean()
        org_logits = classifier(data['feat'], data['sub_adj'])[0][0, data['node_idx_new']]
        alpha_mu = torch.zeros_like(mu)
        alpha_mu[:,:,:args.K] = sample_mu[:,:,:args.K]
        alpha_adj = torch.sigmoid(model.dc(alpha_mu))
        alpha_size = (alpha_adj*data['sub_adj']).sum()
        org_size = data['sub_adj'].sum()
        alpha_sparsity = alpha_size/org_size
        masked_alpha_adj = alpha_adj * data['sub_adj']
        alpha_logits = classifier(data['feat'], masked_alpha_adj)[0][0,data['node_idx_new']]
        return nll_loss, org_logits, alpha_logits, alpha_sparsity

    os.makedirs('explanation/%s' % args.output, exist_ok=True)


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
    else:
        args.retrain = True
        start_epoch = 1
    if args.resume or args.retrain:
        patient = args.patient
        best_loss = 100
        model.train()
        writer = SummaryWriter(comment=args.output)
        start_time = time.time()
        for epoch in tqdm(range(start_epoch, args.epoch+1)):
            batch = 0
            perm = np.random.permutation(num_train)
            train_losses = []
            for beg_ind in range(0, num_train, args.batch_size):
                batch += 1
                end_ind = min(beg_ind+args.batch_size, num_train)
                perm_train_idxs = list(train_idxs[perm[beg_ind: end_ind]])
                optimizer.zero_grad()
                nll_loss, org_logits, alpha_logits, alpha_sparsity = zip(*map(train_task, perm_train_idxs))
                causal_loss = []
                for idx in random.sample(perm_train_idxs, args.NX):
                    _causal_loss, _ = causaleffect.joint_uncond(ceparams, model.dc, classifier, dataset[idx]['sub_adj'], dataset[idx]['feat'], node_idx=dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                    causal_loss += [_causal_loss]
                nll_loss = torch.stack(nll_loss).mean()
                causal_loss = torch.stack(causal_loss).mean()
                alpha_logits = torch.stack(alpha_logits)
                org_logits = torch.stack(org_logits)
                org_probs = F.softmax(org_logits, dim=1)
                klloss = F.kl_div(F.log_softmax(alpha_logits, dim=1), org_probs, reduction='mean')
                alpha_sparsity = torch.stack(alpha_sparsity).mean()
                loss = args.coef_lambda * nll_loss + \
                    args.coef_causal * causal_loss + \
                    args.coef_kl * klloss + \
                    args.coef_size * alpha_sparsity
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                sys.stdout.flush()
                train_losses += [[nll_loss.item(), causal_loss.item(), klloss.item(), alpha_sparsity.item(), loss.item()]]
            nll_loss, causal_loss, klloss, size_loss, train_loss = np.mean(train_losses, axis=0)
            writer.add_scalar("train/nll", nll_loss, epoch)
            writer.add_scalar("train/causal", causal_loss, epoch)
            writer.add_scalar("train/kld(Y_alpha,Y_org)", klloss, epoch)
            writer.add_scalar("train/alpha_sparsity", size_loss, epoch)
            writer.add_scalar("train/total_loss", train_loss, epoch)
            val_loss = eval_model(val_idxs,'val')
            patient -= 1
            if val_loss < best_loss:
                best_loss = val_loss
                patient = 100
                save_checkpoint('explanation/%s/model.ckpt' % args.output)
                eval_model(test_idxs,'test')
            elif patient <= 0:
                print("Early stop.")
                break
        print("Train time:", time.time() - start_time)
        writer.close()
        # Load checkpoint with lowest val loss
        checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print("Start evaluation.")
    model.eval()
    results = []
    with torch.no_grad():
        for idx in tqdm(test_idxs):
            data = dataset[idx]
            org_probs = F.softmax(classifier(data['feat'], data['sub_adj'])[0][:,data['node_idx_new']], dim=1)
            pred_labels = torch.argmax(org_probs, axis=1)
            mu, _ = model.encode(data['sub_feat'], data['adj_norm'])
            alpha_mu = torch.zeros_like(mu)
            alpha_mu[:,:,:args.K] = mu[:,:,:args.K]
            alpha_adj = torch.sigmoid(model.dc(alpha_mu))
            masked_alpha_adj = alpha_adj * data['sub_adj']
            flatten_alpha_adj = masked_alpha_adj.flatten(1)
            for top_k in range(20):
                topk = min(top_k*2, flatten_alpha_adj.shape[-1] - 1)
                topk = torch.tensor([topk], device=device).unsqueeze(-1)
                threshold = torch.gather(flatten_alpha_adj.sort(1,descending=True).values, 1, topk)
                threshold = max(threshold, 1E-6)
                topk_alpha_adj = (flatten_alpha_adj > threshold).float().view(data['sub_adj'].shape)
                alpha_logits = classifier(data['feat'], topk_alpha_adj)[0][:,data['node_idx_new']]
                alpha_probs = F.softmax(alpha_logits, dim=1)
                alpha_log_probs = F.log_softmax(alpha_logits, dim=1)
                results += [{
                    "topk": top_k,
                    "alpha_topk": topk_alpha_adj.sum().item()/2,
                    "alpha_sparsity": (topk_alpha_adj.sum()/data['sub_adj'].sum()).item(),
                    "alpha_gt_acc": (torch.argmax(alpha_probs,axis=1) == label[idx]).float().mean().item(),
                    "alpha_pred_acc": (torch.argmax(alpha_probs,axis=1) == pred_labels).float().mean().item(),
                    "alpha_kld": F.kl_div(alpha_log_probs, org_probs, reduction='batchmean').item()
                }]
    columns = results[0].keys()
    df = pd.DataFrame(results, columns = columns)
    df.to_csv(os.path.join('explanation', args.output, 'results.csv'))
    print(df.groupby('topk').mean())

    if args.plot_info_flow:
        print("Calculating information flow...")
        with torch.no_grad():
            infos = [
                [
                    - causaleffect.joint_uncond_singledim(
                        ceparams, model.dc, classifier,
                        dataset[idx]['sub_adj'], dataset[idx]['feat'],
                        dim, node_idx=dataset[idx]['node_idx_new'],
                        act=torch.sigmoid, device=device
                    )[0] for dim in range(ceparams['z_dim'])
                ] for idx in tqdm(test_idxs)
            ]
            infos = torch.tensor(infos, device=device)
            infos = F.normalize(infos, p=1, dim=1)
            print(infos.mean(0))
        info_flow = []
        for i, graph_idx in enumerate(test_idxs):
            for dim in range(ceparams['z_dim']):
                info_flow += [{
                    'graph_idx': graph_idx,
                    'dim': dim,
                    'info': infos[i,dim].item()
                }]
        columns = info_flow[0].keys()
        df = pd.DataFrame(info_flow, columns = columns)
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
