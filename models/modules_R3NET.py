import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Match(nn.Module):
    def __init__(self, cfg):
        super(Match, self).__init__()
        self.input_dim = cfg.model.change_detector.att_dim
        self.query = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.key = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, q, k):
        simliarity = torch.sigmoid(self.query(q) + self.key(k))
        unchanged = simliarity * k
        changed = q - unchanged
        return changed

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super(SelfAttention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.model.change_detector.att_dim, cfg.model.change_detector.att_head))
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class ChangeDetectorDoubleAttDyn(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim
        self.feat_dim = cfg.model.change_detector.feat_dim
        self.att_head = cfg.model.change_detector.att_head
        self.att_dim = cfg.model.change_detector.att_dim

        self.img = nn.Linear(self.feat_dim, self.att_dim)

        self.un_graph = SelfAttention(cfg)

        self.match = Match(cfg)

        self.fc = nn.Linear(self.att_dim*2, self.att_dim)

        self.dropout = nn.Dropout(0.5)

        self.embed = nn.Sequential(
            nn.Conv2d(self.att_dim * 2, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.fc0 = nn.Linear(self.att_dim*2, self.att_dim)

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.att_dim, 6)

        self.tag1 = nn.Sequential(
            nn.Linear(self.att_dim*3, self.att_dim),
            nn.Dropout(0.5),
            nn.ReLU())
        self.tag2 = nn.Sequential(
            nn.Linear(self.att_dim, 50),
            nn.Dropout(0.5),
            nn.Sigmoid())
        self.tag3 = nn.Linear(50, self.att_dim)
        self.tag4 = nn.Sequential(
            nn.Linear(self.att_dim, self.att_dim),
            nn.Dropout(0.5),
            nn.ReLU())

    def forward(self, input_1, input_2, change=False):
        batch_size, C, H, W = input_1.size()
        input_1 = input_1.view(batch_size, C, -1).permute(0, 2, 1) # (128, 196, 1026)
        input_2 = input_2.view(batch_size, C, -1).permute(0, 2, 1)
        input_bef = torch.relu(self.img(input_1)) # (128,196, 512)
        input_aft = torch.relu(self.img(input_2))
        bef_mask = torch.Tensor(np.ones([batch_size,H*W])).cuda().unsqueeze(1)
        aft_mask = torch.Tensor(np.ones([batch_size,H*W])).cuda().unsqueeze(1)

        input_bef = self.un_graph(input_bef, input_bef, input_bef, bef_mask)
        input_aft = self.un_graph(input_aft, input_aft, input_aft, aft_mask)

        bef_changed = self.match(input_bef, input_aft)
        aft_changed = self.match(input_aft, input_bef)

        changed = self.fc(torch.cat([bef_changed, aft_changed], -1))
        changed = self.dropout(torch.relu(changed))

        input_bef = input_bef.permute(0,2,1).view(batch_size, self.att_dim, H, W)
        input_aft = input_aft.permute(0,2,1).view(batch_size, self.att_dim, H, W)

        changed = changed.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)

        input_global = torch.cat([input_bef, input_aft, changed], 1)
        global_mean = input_global.mean(2).mean(2)

        tag_feat = self.tag1(global_mean)
        tag_pro = self.tag2(tag_feat)
        tag_input = self.dropout(self.tag3(tag_pro))
        tag_input = self.tag4(tag_input)

        input_before = torch.cat([input_bef, changed], 1)
        input_after = torch.cat([input_aft, changed], 1)
        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = torch.sigmoid(self.att(embed_before))
        att_weight_after = torch.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_bef)
        attended_1 = (input_bef * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_aft)
        attended_2 = (input_aft * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended1 = attended_2 - attended_1
        input_attended2 = attended_1 - attended_2
        input_attended = torch.cat([input_attended1, input_attended2], -1)
        input_attended = self.fc0(input_attended)
        input_attended = self.dropout(torch.relu(input_attended))
        pred = self.fc1(input_attended)

        return pred, att_weight_before, att_weight_after, attended_1, attended_2, input_attended, tag_input, tag_pro


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
