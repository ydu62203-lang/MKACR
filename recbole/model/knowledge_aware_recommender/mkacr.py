import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import torch.nn.functional as F
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, logmap0
from abc import ABC, abstractmethod
from typing import Tuple
from utils.euclidean import givens_rotations, givens_reflection

from torch.utils.data import WeightedRandomSampler


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float64)
            self.W2 = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float64)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        norm_matrix = (
            norm_matrix.to(torch.float32)
            if norm_matrix.dtype == torch.float64
            else norm_matrix
        )
        ego_embeddings = (
            ego_embeddings.to(torch.float32)
            if ego_embeddings.dtype == torch.float64
            else ego_embeddings
        )

        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            # Ensure ego_embeddings are of double type within the model definition or forward function
            ego_embeddings = ego_embeddings.to(torch.double)

            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)
        return ego_embeddings


class STUDENT(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(STUDENT, self).__init__(config, dataset)

        # Load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.all_hs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").row
        ).to(self.device)
        self.all_ts = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").col
        ).to(self.device)
        self.all_rs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").data
        ).to(self.device)
        self.matrix_size = torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]
        )

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.layers = [self.embedding_size] + config["layers"]
        self.aggregator_type = config["aggregator_type"]
        self.mess_dropout = config["mess_dropout"]
        self.reg_weight = config["reg_weight"]

        # Generate intermediate data
        self.A_in = (
            self.init_graph()
        )  # Init the attention matrix by the structure of CKG
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in
        self.A_in_3 = self.A_in

        # Projection head for contrastive learning
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = sum(self.layers)
        self.projection_head.append(
            torch.nn.Linear(inner_size, inner_size * 4, bias=False, dtype=torch.float64)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(
                inner_size * 4, eps=1e-12, affine=affine, dtype=torch.float64
            )
        )
        self.projection_head.append(
            torch.nn.Linear(inner_size * 4, inner_size, bias=False, dtype=torch.float64)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(
                inner_size, eps=1e-12, affine=affine, dtype=torch.float64
            )
        )
        self.mode = 0

        # Define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_entity_e"]

        # --- Hyperbolic KG Embedding Parameters ---
        self.data_type = torch.double
        self.sizes = (self.n_entities, self.n_relations, self.n_entities)
        self.rank = self.embedding_size
        self.init_size = 0.001
        self.gamma = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.bh = nn.Embedding(self.n_entities, 1)
        self.bh.weight.data.zero_()
        self.bt = nn.Embedding(self.n_entities, 1)
        self.bt.weight.data.zero_()
        self.new_bt = nn.Embedding(self.n_entities, 1)
        self.new_bh = nn.Embedding(self.n_entities, 1)
        self.new_bh.weight.data.zero_()
        self.new_bt.weight.data.zero_()
        self.entity_embedding.weight.data = self.init_size * torch.randn(
            (self.sizes[0], self.rank), dtype=self.data_type
        )
        self.relation_embedding.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = (
            2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        )
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        self.act = nn.Softmax(dim=1)
        self.regularizer = N3(weight=1.0)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        self.scale = torch.Tensor([1.0 / np.sqrt(self.rank)]).double().to(self.device)
        self.bias = "learn"

        # --- Adversarial and Contrastive Noise Parameters ---
        self.rank = self.embedding_size
        self.noise_u = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_p = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_u_cts1 = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_u_cts2 = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_e_cts1 = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_e_cts2 = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_norm = 0.01
        self.global_counter = 0

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_three_disjoint_indices(self, total_size, subset_size, base_seed=42):
        # Increment the counter each time this function is called
        current_seed = base_seed + self.global_counter
        self.global_counter += 1

        # Set the random seed for reproducibility
        self.set_seed(current_seed)

        all_indices = torch.arange(total_size)
        shuffled_indices = all_indices[torch.randperm(total_size)]

        ui_rand_samples_1 = shuffled_indices[:subset_size]
        ui_rand_samples_2 = shuffled_indices[subset_size : 2 * subset_size]
        ui_rand_samples_3 = shuffled_indices[2 * subset_size : 3 * subset_size]

        return ui_rand_samples_1, ui_rand_samples_2, ui_rand_samples_3

    def init_graph(self):
        r"""Initializes the attention matrix for the collaborative knowledge graph."""
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))

            # A small positive constant to prevent division by zero
            epsilon = 1e-8
            rowsum[rowsum == 0] = epsilon

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.tensor(
            np.vstack((final_adj_matrix.row, final_adj_matrix.col)), dtype=torch.long
        )
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        return torch.cat([user_embeddings, entity_embeddings], dim=0)

    def forward_1(self):
        """Performs a forward pass through the GNN layers using graph view 1."""
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def forward_2(self):
        """Performs a forward pass through the GNN layers using graph view 2."""
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def forward_3(self):
        """Performs a forward pass through the GNN layers using graph view 3."""
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_3, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def forward_1_noise(self):
        """Performs a forward pass with Gaussian noise using graph view 1."""
        ego_embeddings = self._get_ego_embeddings()
        std_dev = 0.01 * torch.ones_like(
            ego_embeddings, dtype=self.data_type, device=ego_embeddings.device
        )
        noise = torch.normal(mean=0.0, std=std_dev)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def forward_2_noise(self):
        """Performs a forward pass with Gaussian noise using graph view 2."""
        ego_embeddings = self._get_ego_embeddings()
        std_dev = 0.01 * torch.ones_like(
            ego_embeddings, dtype=self.data_type, device=ego_embeddings.device
        )
        noise = torch.normal(mean=0.0, std=std_dev)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def forward_3_noise(self):
        """Performs a forward pass with Gaussian noise using graph view 3."""
        ego_embeddings = self._get_ego_embeddings()
        std_dev = 0.01 * torch.ones_like(
            ego_embeddings, dtype=self.data_type, device=ego_embeddings.device
        )
        noise = torch.normal(mean=0.0, std=std_dev)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_3, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        return torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def cts_loss(self, z_i, z_j, temp, batch_size):
        """Calculates the contrastive loss."""
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return self.ce_loss(logits, labels)

    def projection_head_map(self, state, mode):
        for i, l in enumerate(self.projection_head):
            if i % 2 != 0:
                if mode == 0:
                    l.train()  # Set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()  # Set BN to eval mode: use an accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    def improved_sampling(self, embeddings, proportion=0.05):
        """
        Samples indices proportionally from the embedding matrix.
        Can be extended to incorporate randomness and hard-negative mining.

        Args:
            embeddings (torch.Tensor): The input embedding matrix.
            proportion (float): The proportion of samples to draw.

        Returns:
            list: A list of sampled indices.
        """
        num_samples = int(embeddings.shape[0] * proportion)
        weights = torch.ones(embeddings.shape[0])
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=False
        )
        return list(sampler)

    def _calculate_adversarial_loss(
        self, interaction, forward_func, forward_noise_func
    ):
        """A helper function to compute recommendation and adversarial contrastive loss."""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Standard forward pass for recommendation loss
        user_all_embeddings, entity_all_embeddings = forward_func()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # Contrastive learning with Gaussian noise
        user_all_embeddings_n, entity_all_embeddings_n = forward_noise_func()
        u_embeddings_n = user_all_embeddings_n[user]
        pos_embeddings_n = entity_all_embeddings_n[pos_item]

        u_embeddings_n_proj = self.projection_head_map(u_embeddings_n, self.mode)
        pos_embeddings_n_proj = self.projection_head_map(
            pos_embeddings_n, 1 - self.mode
        )

        ui_cts_loss_n = self.cts_loss(
            u_embeddings_n_proj,
            pos_embeddings_n_proj,
            temp=1.0,
            batch_size=u_embeddings_n.shape[0],
        )

        # Generate adversarial noise
        u_embeddings_n_proj.retain_grad()
        pos_embeddings_n_proj.retain_grad()
        ui_cts_loss_n.backward(retain_graph=True)

        u_grad = u_embeddings_n_proj.grad.clone().detach()
        p_grad = pos_embeddings_n_proj.grad.clone().detach()

        # Normalize the noise
        self.noise_u = self.noise_norm * F.normalize(u_grad, p=2, dim=-1)
        self.noise_p = self.noise_norm * F.normalize(p_grad, p=2, dim=-1)

        # Adversarial contrastive learning
        gan_u_cts_embedding = u_embeddings_n[user] + self.noise_u
        gan_p_cts_embedding = entity_all_embeddings_n[pos_item] + self.noise_p

        gan_u_cts_embedding_proj = self.projection_head_map(
            gan_u_cts_embedding, self.mode
        )
        gan_p_cts_embedding_proj = self.projection_head_map(
            gan_p_cts_embedding, 1 - self.mode
        )

        # Calculate the contrastive loss with adversarial noise
        ui_cts_loss_adv = self.cts_loss(
            gan_u_cts_embedding_proj,
            gan_p_cts_embedding_proj,
            temp=1.0,
            batch_size=gan_u_cts_embedding.shape[0],
        )

        # Switch the projection head mode for the next iteration
        self.mode = 1 - self.mode

        return mf_loss + self.reg_weight * reg_loss + 0.01 * ui_cts_loss_adv

    def calculate_o_loss(self, interaction):
        """Calculates the loss for the 'original' graph view."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None
        return self._calculate_adversarial_loss(
            interaction, self.forward_2, self.forward_2_noise
        )

    def calculate_h_loss(self, interaction):
        """Calculates the loss for the 'hyperbolic' graph view."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None
        return self._calculate_adversarial_loss(
            interaction, self.forward_1, self.forward_1_noise
        )

    def calculate_e_loss(self, interaction):
        """Calculates the loss for the 'Euclidean' graph view."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None
        return self._calculate_adversarial_loss(
            interaction, self.forward_3, self.forward_3_noise
        )

    def get_rhs(self, queries):
        """Get embeddings and biases of target entities."""
        return self.entity_embedding(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores for queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return -hyp_distance_multi_c(lhs_e, rhs_e, c) ** 2

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity_embedding(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1).to(self.device)
        context_vec = (
            self.context_vec(queries[:, 1]).view((-1, 1, self.rank)).to(self.device)
        )
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel = self.relation_embedding(queries[:, 1])
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

    def score(self, lhs, rhs):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e)
        if self.bias == "constant":
            return self.gamma.item() + score
        elif self.bias == "learn":
            return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization."""
        head_e = self.entity_embedding(queries[:, 0])
        rel_e = self.relation_embedding(queries[:, 1])
        rhs_e = self.entity_embedding(queries[:, 2])
        return head_e, rel_e, rhs_e

    def Forward(self, queries):
        """KGModel forward pass."""
        lhs_e, lhs_biases = self.get_queries(queries)
        rhs_e, rhs_biases = self.get_rhs(queries)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))
        factors = self.get_factors(queries)
        return predictions, factors

    def neg_sampling_loss(self, input_batch, neg_samples):
        """Compute KG embedding loss with negative sampling."""
        positive_score, factors = self.Forward(input_batch)
        positive_score = F.logsigmoid(positive_score)
        negative_score, _ = self.Forward(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = -torch.cat([positive_score, negative_score], dim=0).mean()
        return loss, factors

    def calculate_loss1(self, input_batch, neg_samples):
        """Compute KG embedding loss and regularization loss."""
        loss, factors = self.neg_sampling_loss(input_batch, neg_samples)
        loss += self.regularizer.forward(factors)
        return loss

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )
        h_e = torch.bmm(h_e.double(), r_trans_w.double()).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e.double(), r_trans_w.double()).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e.double(), r_trans_w.double()).squeeze(1)
        return h_e, r_e, pos_t_e, neg_t_e

    def _get_rotate_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h)
        pos_t_e = self.entity_embedding(pos_t)
        neg_t_e = self.entity_embedding(neg_t)
        r_e = self.relation_embedding(r)

        h_real, h_imag = torch.chunk(h_e, 2, dim=-1)
        pos_t_real, pos_t_imag = torch.chunk(pos_t_e, 2, dim=-1)
        neg_t_real, neg_t_imag = torch.chunk(neg_t_e, 2, dim=-1)
        r_real, r_imag = torch.chunk(r_e, 2, dim=-1)

        head_e = (h_real, h_imag)
        pos_rhs_e = (pos_t_real, pos_t_imag)
        neg_rhs_e = (neg_t_real, neg_t_imag)
        rel_e = (r_real, r_imag)

        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = (head_e[0] * cos - head_e[1] * sin, head_e[0] * sin + head_e[1] * cos)
        return lhs_e, rel_e, pos_rhs_e, neg_rhs_e

    def calculate_kg_h_loss(self, interaction):
        """Calculates the KG loss for the hyperbolic view (AttH/RotH)."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        input_batch = torch.stack((h, r, pos_t), dim=1)
        neg_samples = torch.stack((h, r, neg_t), dim=1)
        return self.calculate_loss1(input_batch, neg_samples)

    def calculate_kg_o_loss(self, interaction):
        """Calculates the KG loss for the original view (TransE)."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = torch.sum((h_e + r_e - pos_t_e) ** 2, dim=1)
        neg_tail_score = torch.sum((h_e + r_e - neg_t_e) ** 2, dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        return kg_loss + self.reg_weight * kg_reg_loss

    def calculate_kg_e_loss(self, interaction):
        """Calculates the KG loss for the Euclidean view (RotatE)."""
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        lhs_e, r_e, pos_t_e, neg_t_e = self._get_rotate_embedding(h, r, pos_t, neg_t)
        pos_tail_score = torch.sum(
            lhs_e[0] * pos_t_e[0] + lhs_e[1] * pos_t_e[1], 1, keepdim=True
        )
        neg_tail_score = torch.sum(
            lhs_e[0] * neg_t_e[0] + lhs_e[1] * neg_t_e[1], 1, keepdim=True
        )
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(
            lhs_e[0],
            r_e[0],
            pos_t_e[0],
            neg_t_e[0],
            lhs_e[1],
            r_e[1],
            pos_t_e[1],
            neg_t_e[1],
        )
        return kg_loss + self.reg_weight * kg_reg_loss

    def generate_transE_score3(self, hs, ts, r):
        """Calculates KG triple scores based on a RotatE-like scoring function."""
        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding(r)

        head_e_real, head_e_imag = torch.chunk(h_e, 2, dim=-1)
        tail_e_real, tail_e_imag = torch.chunk(t_e, 2, dim=-1)
        rel_e_real, rel_e_imag = torch.chunk(r_e, 2, dim=-1)

        rel_norm = torch.sqrt(rel_e_real**2 + rel_e_imag**2)
        cos = rel_e_real / rel_norm
        sin = rel_e_imag / rel_norm

        lhs_e_real = head_e_real * cos - head_e_imag * sin
        lhs_e_imag = head_e_real * sin + head_e_imag * cos

        score = torch.mul(lhs_e_real, self.tanh(tail_e_real)).sum(dim=1) + torch.mul(
            lhs_e_imag, self.tanh(tail_e_imag)
        ).sum(dim=1)
        return score

    def generate_transE_score1(self, hs, ts, r):
        """Calculates KG triple scores based on a hyperbolic scoring function."""
        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding(r)

        c = F.softplus(self.c[r])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(r), 2, dim=1)
        rot_q = givens_rotations(rot_mat, h_e).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, h_e).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)

        rel = expmap0(r_e, c)
        res = project(mobius_add(lhs, rel, c), c)

        res_euclidean = logmap0(res, c)
        t_e_euclidean = logmap0(t_e, c)

        kg_score = torch.mul(t_e_euclidean, self.tanh(res_euclidean)).sum(dim=1)
        return kg_score

    def generate_transE_score2(self, hs, ts, r):
        """Calculates KG triple scores based on a TransE-like scoring function."""
        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        )

        h_e_proj = torch.matmul(h_e.double(), r_trans_w.double())
        t_e_proj = torch.matmul(t_e.double(), r_trans_w.double())

        kg_score = torch.mul(t_e_proj, self.tanh(h_e_proj + r_e)).sum(dim=1)
        return kg_score

    def update_attentive_A(self):
        """Updates the attention matrices based on the current embeddings."""
        kg_score_list_1, kg_score_list_2, kg_score_list_3, row_list, col_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)[0]
            if len(triple_index) == 0:
                continue

            hs_rel = self.all_hs[triple_index]
            ts_rel = self.all_ts[triple_index]
            rs_rel = self.all_rs[triple_index]

            kg_score1 = self.generate_transE_score1(hs_rel, ts_rel, rs_rel)
            kg_score2 = self.generate_transE_score2(hs_rel, ts_rel, rel_idx)
            kg_score3 = self.generate_transE_score3(hs_rel, ts_rel, rs_rel)

            row_list.append(hs_rel)
            col_list.append(ts_rel)
            kg_score_list_1.append(kg_score1)
            kg_score_list_2.append(kg_score2)
            kg_score_list_3.append(kg_score3)

        if not row_list:
            return

        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.stack([row, col], dim=0)

        # Move to CPU for sparse softmax, which is not supported on CUDA sparse tensors
        kg_score1 = torch.cat(kg_score_list_1, dim=0)
        A_in_1 = torch.sparse.FloatTensor(indices, kg_score1, self.matrix_size).cpu()
        self.A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)

        kg_score2 = torch.cat(kg_score_list_2, dim=0)
        A_in_2 = torch.sparse.FloatTensor(indices, kg_score2, self.matrix_size).cpu()
        self.A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)

        kg_score3 = torch.cat(kg_score_list_3, dim=0)
        A_in_3 = torch.sparse.FloatTensor(indices, kg_score3, self.matrix_size).cpu()
        self.A_in_3 = torch.sparse.softmax(A_in_3, dim=1).to(self.device)

    def predict(self, interaction):
        """
        Predicts the scores for given user-item pairs.

        Args:
            interaction (Interaction): Contains user_id and item_id.

        Returns:
            torch.Tensor: The prediction scores.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()
        user_all_embeddings_3, entity_all_embeddings_3 = self.forward_3()

        u_embeddings_1 = user_all_embeddings_1[user]
        i_embeddings_1 = entity_all_embeddings_1[item]
        u_embeddings_2 = user_all_embeddings_2[user]
        i_embeddings_2 = entity_all_embeddings_2[item]
        u_embeddings_3 = user_all_embeddings_3[user]
        i_embeddings_3 = entity_all_embeddings_3[item]

        scores_1 = torch.mul(u_embeddings_1, i_embeddings_1).sum(dim=1)
        scores_2 = torch.mul(u_embeddings_2, i_embeddings_2).sum(dim=1)
        scores_3 = torch.mul(u_embeddings_3, i_embeddings_3).sum(dim=1)

        return scores_1 + scores_2 + scores_3

    def full_sort_predict(self, interaction):
        """
        Calculates scores for a user against all items for full ranking.

        Args:
            interaction (Interaction): Contains user_id.

        Returns:
            torch.Tensor: A tensor of scores for all items.
        """
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e_1, self.restore_entity_e_1 = self.forward_1()
            self.restore_user_e_2, self.restore_entity_e_2 = self.forward_2()
            self.restore_user_e_3, self.restore_entity_e_3 = self.forward_3()

        u_embeddings_1 = self.restore_user_e_1[user]
        i_embeddings_1 = self.restore_entity_e_1[: self.n_items]
        u_embeddings_2 = self.restore_user_e_2[user]
        i_embeddings_2 = self.restore_entity_e_2[: self.n_items]
        u_embeddings_3 = self.restore_user_e_3[user]
        i_embeddings_3 = self.restore_entity_e_3[: self.n_items]

        scores_1 = torch.matmul(u_embeddings_1, i_embeddings_1.transpose(0, 1))
        scores_2 = torch.matmul(u_embeddings_2, i_embeddings_2.transpose(0, 1))
        scores_3 = torch.matmul(u_embeddings_3, i_embeddings_3.transpose(0, 1))

        return (scores_1 + scores_2 + scores_3).view(-1)
