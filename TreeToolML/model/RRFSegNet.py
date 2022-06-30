import torch.nn.functional as F
import torch.nn as nn
import torch
from .build_arch import ARCH_REGISTRY


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


if device == "cuda":
    torch.backends.cudnn.benchmark = True


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


def build_conv_block(input_shape, filters, use_dropout=False):
    conv_block = []
    conv_block += (
        nn.Conv2d(input_shape, filters, kernel_size=[1, 1], stride=[1, 1], padding=0),
    )
    conv_block += (nn.BatchNorm2d(filters, affine=False),)
    conv_block += (nn.ReLU(),)
    if use_dropout:
        conv_block += (nn.Dropout(),)
    return nn.Sequential(*conv_block)


class relation_reasoning_layers(nn.Module):
    def __init__(self, input_shape, nodes_list):
        super(relation_reasoning_layers, self).__init__()
        self.block1 = nn.Sequential(
            build_conv_block(input_shape, nodes_list[0]),
            build_conv_block(nodes_list[0], nodes_list[1]),
        )
        self.block1.apply(init_weights)
        self.block2 = nn.Sequential(build_conv_block(nodes_list[1], nodes_list[2]))
        self.block2.apply(init_weights)

    def forward(self, x):
        x = self.block1(x)
        x, _ = torch.max(x, dim=2, keepdim=True)
        x = self.block2(x)
        return x


def pairwise_distance(point_cloud):
    return torch.cdist(point_cloud, point_cloud) ** 2


def knn(adj_matrix, k=5):
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k=k + 1)
    return nn_idx[:, :, 1:]


def get_relation_features(point_features, nn_idx, k):
    #output batch, points, k features, [original point, n nearest neighbor, n+1 nearest neighbor]
    #prepare data
    og_batch_size = point_features.shape[0]
    point_features = torch.squeeze(point_features)
    if og_batch_size == 1:
        point_features = torch.unsqueeze(point_features, 0)

    batch_size = point_features.shape[0]
    channels = point_features.shape[1]

    # get neigbors from nn_idx
    point_cloud_central = point_features
    point_cloud_neighbors = point_features[torch.arange(batch_size)[:,None,None,None],torch.arange(channels)[None,:,None,None],nn_idx[:,None,:,:]]
    point_cloud_central = torch.unsqueeze(point_cloud_central, dim=-2)

    #get neigbors relative to point
    point_cloud_central = point_cloud_central.repeat(1, 1, k, 1)
    point_cloud_neighbors = point_cloud_neighbors - point_cloud_central


    num_vertex_pairs = k
    vertex_pairs_list = [(i, i + 1) for i in range(k - 1)]
    vertex_pairs_list.append((k - 1, 0))
    for i in range(num_vertex_pairs):

        temp_concat = torch.cat(
            [
                point_cloud_neighbors[:, :,vertex_pairs_list[i][0],:],
                point_cloud_neighbors[:, :,vertex_pairs_list[i][1],:],
            ],
            dim=-2,
        )

        temp_concat = torch.unsqueeze(temp_concat, -2)
        if i == 0:
            relation_features = temp_concat
        else:
            relation_features = torch.cat([relation_features, temp_concat], dim=-2)

    point_features = torch.unsqueeze(point_features, dim=-2)
    point_features = point_features.repeat(1, 1, num_vertex_pairs, 1)
    relation_features = torch.cat([point_features, relation_features], dim=1)
    return relation_features

@ARCH_REGISTRY.register("RRFSegNet")
class get_model_RRFSegNet(nn.Module):
    def __init__(self, MODEL_CFG):
        super(get_model_RRFSegNet, self).__init__()
        self.net_1 = relation_reasoning_layers(9, nodes_list=[64, 64, 64])
        self.net_2 = relation_reasoning_layers(192, nodes_list=[128, 128, 128])
        self.global_net = nn.Conv2d(
            192, 1024, kernel_size=[1, 1], stride=[1, 1], padding=0
        )
        self.end_net = nn.Sequential(
            build_conv_block(1216, 256, True),
            build_conv_block(256, 64),
            nn.Conv2d(64, MODEL_CFG.OUTPUT_NODS, kernel_size=[1, 1], stride=[1, 1], padding=0),
            nn.BatchNorm2d(MODEL_CFG.OUTPUT_NODS, affine=False),
        )

    def forward(self, x):
        points = x
        num_point = points.shape[1]
        Position = points[:, :, :3]
        adj = pairwise_distance(Position)
        nn_idx = knn(adj, k=20)

        points = points.permute(0,2,1)
        nn_idx = nn_idx.permute(0,2,1)
        relation_features1 = get_relation_features(points, nn_idx=nn_idx, k=20)
        out_net1 = self.net_1(relation_features1)

        relation_features2 = get_relation_features(out_net1, nn_idx=nn_idx, k=20)
        out_net2 = self.net_2(relation_features2)

        global_net_in = torch.cat([out_net1, out_net2], dim=1)
        global_net_out = self.global_net(global_net_in)
        global_net, _ = torch.max(global_net_out, dim=-1, keepdim=True)
        global_net = global_net.repeat([1, 1, 1, num_point])
        concat = torch.cat([global_net, out_net1, out_net2], dim=1)
        out = self.end_net(concat)

        og_batch_size = out.shape[0]
        out = torch.squeeze(out)
        if og_batch_size == 1:
            out = torch.unsqueeze(out, 0)
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
            concat = tf.concat(axis=DeepPointwiseDirections)
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
