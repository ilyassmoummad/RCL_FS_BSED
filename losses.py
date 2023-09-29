import torch
from torch import nn
from torch.nn import functional as F
from args import args

class SupConLoss(nn.Module): # from : https://github.com/ilyassmoummad/scl_icbhi2017/blob/main/losses.py
    def __init__(self, temperature=0.06, device="cuda:0", usetcr=True): # temperature was not explored for this task
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.tcr_fn = TotalCodingRate(eps=args.eps)
        self.use_tcr = usetcr

    def forward(self, projection1, projection2, labels=None):

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        #log_prob = logits - torch.log((exp_logits - (logits * mask)).sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        if self.use_tcr:
            loss += args.alpha * ( self.tcr_fn(projection1) + self.tcr_fn(projection2) ) / 2  
        return loss

class ProtoCLR(nn.Module):
    def __init__(self, tau=1.):
        super(ProtoCLR, self).__init__()
        self.tau = tau
        self.tcr_fn = TotalCodingRate(eps=args.eps)

    #def forward(self, h1_features, h2_features, z1_features, z2_features, labels):
    def forward(self, z1_features, z2_features, labels):

        labels = torch.cat([labels, labels], dim=0)

        #h1_features, h2_features = h1_features.detach(), h2_features.detach()

        z1_features, z2_features = F.normalize(z1_features), F.normalize(z2_features)
        #h1_features, h2_features = F.normalize(h1_features), F.normalize(h2_features)

        #h_features = torch.cat([h1_features, h2_features])
        z_features = torch.cat([z1_features, z2_features])

        unique_labels = torch.unique(labels)

        # Compute similarity between each feature and its corresponding class mean
        feature_means = torch.stack([z_features[labels == label].mean(dim=0) for label in unique_labels])

        feature_means_repeated = torch.zeros((z_features.shape[0], z_features.shape[1])).to(z_features.device)
        for label in torch.unique(labels):
            feature_means_repeated[labels==label] = torch.mean(z_features[labels==label], dim=0)

        sim_proto = torch.diag(torch.mm(feature_means_repeated, z_features.T)) / self.tau

        sim_all = torch.mm(z_features, feature_means.T) / self.tau

        # Formulate the loss as NT-Xent
        #exp_sim = torch.exp(sim_all)
        exp_sim = torch.exp(sim_all - sim_proto.unsqueeze(1).repeat(1, sim_all.shape[1]))
        log_prob_pos = sim_proto - torch.log(exp_sim.sum(1))
        loss = - log_prob_pos.mean()

        #loss += args.alpha * ( self.tcr_fn(z1_features) + self.tcr_fn(z2_features) ) / 2

        return loss#.mean()

class TotalCodingRate(nn.Module):
    def __init__(self, eps=1.):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)

class TCRLoss(nn.Module):
    def __init__(self, tcr_coeff=1e-4, eps=0.1):
        super(TCRLoss, self).__init__()
        self.tcr_fn = TotalCodingRate(eps=eps)
        self.tcr_coeff = tcr_coeff

    def forward(self, z1_1, z1_2, z2_1, z2_2, h1_features, h2_features, z1_features, z2_features, labels):

        batch_size = h1_features.shape[0]
        emb_size = h1_features.shape[1]

        h1_features, h2_features = h1_features.detach(), h2_features.detach()  

        #h1_features, h2_features = F.normalize(h1_features), F.normalize(h2_features)
        #z1_features, z2_features = F.normalize(z1_features), F.normalize(z2_features)
        #z1_1, z1_2, z2_1, z2_2 = F.normalize(z1_1), F.normalize(z1_2), F.normalize(z2_1), F.normalize(z2_2)
        
        h1_proto = torch.zeros((batch_size, emb_size)).to(args.device)
        h2_proto = torch.zeros((batch_size, emb_size)).to(args.device)
        for label in torch.unique(labels):
            h1_proto[labels==label] = torch.mean(h1_features[labels==label], dim=0)
            h2_proto[labels==label] = torch.mean(h2_features[labels==label], dim=0)

        sim1 = F.cosine_similarity(z1_features, h2_proto, dim=-1)
        sim2 = F.cosine_similarity(z2_features, h1_proto, dim=-1)

        proto_loss = - 0.5 * ( sim1.mean() + sim2.mean() )

        #cov_loss = self.cov_fn(z1_features, z2_features) #+ self.cov_fn(z2_features, h1_proto) ) / 2

        #tcr_loss = ( self.tcr_fn(z1_features) + self.tcr_fn(z2_features) ) / 2 #only last
        tcr_loss = ( ( self.tcr_fn(z1_features) + self.tcr_fn(z2_features) ) + 0.5 * ( self.tcr_fn(z2_1) + self.tcr_fn(z2_2) ) + 0.25 * ( self.tcr_fn(z1_1) + self.tcr_fn(z1_2) ) ) / 2

        #var_loss = self.var_fn(z1_features, z2_features)

        #loss = self.inv_coeff * proto_loss + self.cov_coeff * cov_loss + self.var_coeff * var_loss

        loss = proto_loss + self.tcr_coeff * tcr_loss

        return loss