"""
In theory, the backbone network can be any semantic segmentation
framework that directly takes discrete points as input.

Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import torch.nn.functional as F
import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


if device == "cuda":
    torch.backends.cudnn.benchmark = True


class relation_reasoning_layers(nn.Module):
    def __init__(self, input_shape, nodes_list):
        super(relation_reasoning_layers, self).__init__()
        self.conv1 = nn.Conv2d(
            input_shape, nodes_list[0], kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(nodes_list[0], affine=False)
        self.conv2 = nn.Conv2d(
            nodes_list[0], nodes_list[1], kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(nodes_list[1], affine=False)

        self.conv3 = nn.Conv2d(
            nodes_list[1], nodes_list[2], kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(nodes_list[2], affine=False)

    def forward(self, x):
        x = x.permute([0, 3, 2, 1])
        x = F.relu(self.bn1(self.conv1(x)))
        x, _ = torch.max(F.relu(self.bn2(self.conv2(x))), dim=2, keepdim=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute([0, 3, 2, 1])
        return x


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
        point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    return torch.cdist(point_cloud, point_cloud) ** 2


def knn(adj_matrix, k=5):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k=k + 1)  ### remove the current point
    return nn_idx[:, :, 1:]


def get_relation_features(point_features, nn_idx, k):
    og_batch_size = point_features.shape[0]
    point_features = torch.squeeze(point_features)
    if og_batch_size == 1:
        point_features = torch.unsqueeze(point_features, 0)

    point_cloud_central = point_features

    point_cloud_shape = point_features.shape
    batch_size = point_cloud_shape[0]
    num_points = point_cloud_shape[1]
    num_dims = point_cloud_shape[2]

    idx_ = torch.range(0, batch_size - 1, device=device) * num_points
    idx_ = torch.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = torch.reshape(point_features, [-1, num_dims])
    point_cloud_neighbors = point_cloud_flat[(nn_idx + idx_).long()]
    point_cloud_central = torch.unsqueeze(point_cloud_central, dim=-2)

    point_cloud_central = point_cloud_central.repeat(1, 1, k, 1)
    point_cloud_neighbors = point_cloud_neighbors - point_cloud_central

    #############
    # rank_state_sums = int(comb(k, 2))
    # list_rank_state = list(combinations(list(range(k)), 2))
    #############

    num_vertex_pairs = k
    vertex_pairs_list = [(i, i + 1) for i in range(k - 1)]
    vertex_pairs_list.append((k - 1, 0))
    for i in range(num_vertex_pairs):

        temp_concat = torch.cat(
            [
                point_cloud_neighbors[:, :, vertex_pairs_list[i][0], :],
                point_cloud_neighbors[:, :, vertex_pairs_list[i][1], :],
            ],
            dim=-1,
        )

        temp_concat = torch.unsqueeze(temp_concat, -2)
        if i == 0:
            relation_features = temp_concat
        else:
            relation_features = torch.cat([relation_features, temp_concat], dim=-2)

    point_features = torch.unsqueeze(point_features, dim=-2)
    point_features = point_features.repeat(1, 1, num_vertex_pairs, 1)
    relation_features = torch.cat([point_features, relation_features], dim=-1)
    return relation_features


class get_model_RRFSegNet(nn.Module):
    def __init__(self):
        super(get_model_RRFSegNet, self).__init__()
        self.net_1 = relation_reasoning_layers(9, nodes_list=[64, 64, 64])
        ### layer_2
        self.net_2 = relation_reasoning_layers(192, nodes_list=[128, 128, 128])
        ###generate global features
        self.global_net = nn.Conv2d(
            192, 1024, kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        self.end_net1 = nn.Conv2d(
            1216, 256, kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        torch.nn.init.xavier_uniform(self.end_net1.weight)
        self.end_bn1 = nn.BatchNorm2d(256, affine=False)
        self.drop_out = nn.Dropout(p=0.4)
        self.end_net2 = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=[1, 1], padding=0)
        torch.nn.init.xavier_uniform(self.end_net2.weight)
        self.end_bn2 = nn.BatchNorm2d(64, affine=False)
        self.end_net3 = nn.Conv2d(64, 3, kernel_size=[1, 1], stride=[1, 1], padding=0)
        torch.nn.init.xavier_uniform(self.end_net3.weight)
        self.end_bn3 = nn.BatchNorm2d(3, affine=False)

    def forward(self, x):
        points = x
        num_point = points.shape[1]
        Position = points[:, :, :3]
        adj = pairwise_distance(Position)
        nn_idx = knn(adj, k=20)

        relation_features1 = get_relation_features(points, nn_idx=nn_idx, k=20)
        out_net1 = self.net_1(relation_features1)

        relation_features2 = get_relation_features(out_net1, nn_idx=nn_idx, k=20)
        out_net2 = self.net_2(relation_features2)

        global_net_in = torch.cat([out_net1, out_net2], dim=3)
        global_net_in = global_net_in.permute(0, 3, 2, 1)
        global_net_out = self.global_net(global_net_in)
        global_net, _ = torch.max(global_net_out, dim=-1, keepdim=True)
        global_net = global_net.repeat([1, 1, 1, num_point])
        global_net = global_net.permute(0, 3, 2, 1)
        concat = torch.cat([global_net, out_net1, out_net2], dim=3)
        concat = concat.permute(0, 3, 2, 1)
        out = F.relu(self.end_bn1(self.end_net1(concat)))
        out = self.drop_out(out)
        out = F.relu(self.end_bn2(self.end_net2(out)))
        out = self.end_bn3(self.end_net3(out))
        out = torch.squeeze(out)
        return out


if False:

    def get_model_RRFSegNet(
        name,
        points,
        is_training,
        k=20,
        is_dist=True,
        weight_decay=0.0004,
        bn_decay=None,
        reuse=tf.AUTO_REUSE,
    ):
        """RRFSegNet-based Backbone Network (PDE-net)"""

        with tf.variable_scope(name, reuse=reuse):
            num_point = points.get_shape()[1].value
            Position = points[:, :, :3]
            adj = tf_util.pairwise_distance(Position)
            nn_idx = tf_util.knn(adj, k=k)
            ### layer_1
            relation_features1 = tf_util.get_relation_features(
                points, nn_idx=nn_idx, k=k
            )
            net_1 = relation_reasoning_layers(
                "layer_1",
                relation_features1,
                is_training=is_training,
                bn_decay=bn_decay,
                nodes_list=[64, 64, 64],
                weight_decay=weight_decay,
                is_dist=is_dist,
            )
            ### layer_2
            relation_features1 = tf_util.get_relation_features(
                net_1, nn_idx=nn_idx, k=k
            )
            net_2 = relation_reasoning_layers(
                "layer_2",
                relation_features1,
                is_training=is_training,
                bn_decay=bn_decay,
                nodes_list=[128, 128, 128],
                weight_decay=weight_decay,
                is_dist=is_dist,
            )

            ###generate global features
            global_net = tf_util.conv2d(
                tf.concat([net_1, net_2], axis=-1),
                1024,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                weight_decay=weight_decay,
                bn=True,
                is_training=is_training,
                scope="mpl_global",
                bn_decay=bn_decay,
                is_dist=is_dist,
            )

            global_net = tf.reduce_max(global_net, axis=1, keep_dims=True)
            global_net = tf.tile(global_net, [1, num_point, 1, 1])

            ###
            concat = tf.concat(axis=3, values=[global_net, net_1, net_2])

            # CONV
            net = tf_util.conv2d(
                concat,
                256,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                scope="dir/conv1",
                weight_decay=weight_decay,
                is_dist=is_dist,
                bn_decay=bn_decay,
            )
            net = tf_util.dropout(
                net, keep_prob=0.7, is_training=is_training, scope="dp1"
            )
            net = tf_util.conv2d(
                net,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                scope="dir/conv2",
                is_dist=is_dist,
            )
            net = tf_util.conv2d(
                net,
                3,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                activation_fn=None,
                is_training=is_training,
                scope="dir/conv3",
                is_dist=is_dist,
            )
            net = tf.squeeze(net, axis=2)

            return net

    def get_model_DGCNN(
        name,
        point_cloud,
        is_training,
        is_dist=False,
        weight_decay=0.0001,
        bn_decay=None,
        k=20,
        reuse=tf.AUTO_REUSE,
    ):
        """DGCNN-based backbone network (PDE-net)"""

        with tf.variable_scope(name, reuse=reuse):

            num_point = point_cloud.get_shape()[1].value
            input_image = tf.expand_dims(point_cloud, -1)
            input_point_cloud = tf.expand_dims(point_cloud, -2)
            adj = tf_util.pairwise_distance(point_cloud[:, :, :3])
            nn_idx = tf_util.knn(adj, k=k)
            ###
            edge_feature1 = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
            net = tf_util.conv2d(
                edge_feature1,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                weight_decay=weight_decay,
                scope="adj_conv1",
                bn_decay=bn_decay,
                is_dist=is_dist,
            )
            net_1 = tf.reduce_max(net, axis=-2, keep_dims=True)

            edge_feature2 = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)
            net = tf_util.conv2d(
                edge_feature2,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                weight_decay=weight_decay,
                scope="adj_conv3",
                bn_decay=bn_decay,
                is_dist=is_dist,
            )
            net_2 = tf.reduce_max(net, axis=-2, keep_dims=True)

            edge_feature3 = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)
            net = tf_util.conv2d(
                edge_feature3,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                weight_decay=weight_decay,
                scope="adj_conv5",
                bn_decay=bn_decay,
                is_dist=is_dist,
            )
            net_3 = tf.reduce_max(net, axis=-2, keep_dims=True)

            net = tf_util.conv2d(
                tf.concat([net_1, net_2, net_3], axis=-1),
                1024,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                scope="adj_conv7",
                bn_decay=bn_decay,
                is_dist=is_dist,
            )
            out_max = tf_util.max_pool2d(
                net, [num_point, 1], padding="VALID", scope="maxpool"
            )
            expand = tf.tile(out_max, [1, num_point, 1, 1])

            ##############
            net = tf.concat(
                axis=3, values=[expand, net_1, net_2, net_3, input_point_cloud]
            )
            ############
            net = tf_util.conv2d(
                net,
                512,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                scope="dir/conv1",
                is_dist=is_dist,
            )
            net = tf_util.dropout(
                net, keep_prob=0.7, is_training=is_training, scope="dp1"
            )
            net = tf_util.conv2d(
                net,
                64,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                is_training=is_training,
                scope="dir/conv2",
                is_dist=is_dist,
            )
            net = tf_util.dropout(
                net, keep_prob=0.7, is_training=is_training, scope="dp2"
            )
            net = tf_util.conv2d(
                net,
                3,
                [1, 1],
                padding="VALID",
                stride=[1, 1],
                bn=True,
                activation_fn=None,
                is_training=is_training,
                scope="dir/conv3",
                is_dist=is_dist,
            )
            net = tf.squeeze(net, axis=2)
            return net
