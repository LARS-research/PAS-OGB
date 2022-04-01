from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# NA_PRIMITIVES = ['gin']

# NA_graphnas
# NA_PRIMITIVES = [
#   'sage',
#   'gcn',
#   'gin',
#   'gat',
#   'sage_sum',
#   'sage_max',
#   'gat_sym',
#   'gat_cos',
#   'gat_linear',
#   'gat_generalized_linear',
#   #####
# ]

NA_PRIMITIVES = [
  # 'sage',

  # 'gcn',
  # 'gin',

  'gcn_mol',
  'gin_mol',
  'gat',
  # 'gine',
  'mf',
  'Transformerconv',
  'gen',
  'gatv2',
  'gineplus',
  # 'graphconv_add',  # aggr:add mean max
  # 'mlp',



  ########reduced2
  # 'geniepath',
  # 'sgc',
  # 'leconv',
  # ###reduced
  # 'graphconv_mean',
  # 'graphconv_max',
  # 'sage_sum',
  # 'sage_max',
  # 'gat_sym',
  # 'gat_cos',
  # 'gat_linear',
  # 'gat_generalized_linear',
  #####
#
]

NA_MLP_PRIMITIVES = []
for w in [8, 16, 32, 64]:
    for d in [1, 2, 3]:
        NA_MLP_PRIMITIVES.append('mlp_%d_%d' % (w, d))
NA_PRIMITIVES2 = [
  'sage_5',
  'sage_10',
  'sage_15',
  'sage_20',
  'sage_25',
  #'gcn_5',
  #'gcn_10',
  #'gcn_15',
  #'gcn_20',
  #'gcn_25',
  #'gin_5',
  #'gin_10',
  #'gin_15',
  #'gin_20',
  #'gin_25',
  'gat_5',
  'gat_10',
  'gat_15',
  'gat_20',
  'gat_25',
]
POOL_PRIMITIVES=[
  #base on structures
  'hoppool_1',
  'hoppool_2',
  'hoppool_3',

  #based on features.
  'topkpool',
  'mlppool',

    #based on the gap between neighbors and itself.
  'gappool',

  #based on the summarmary between neighbors and itself.
  'sagpool',#GNN=GCN/GAT/graphconv/SAGE
  'asappool',
  'sag_graphconv',


  # 'edgepool',
 #GCN/GraphCConv
  # 'asappool'
  # 'diffgcn',
  # 'diffgraphconv',
  # 'diffsage',
  'none',
  # 'none2',
  # 'none3',
]
READOUT_PRIMITIVES = [
  'global_mean',
 'global_max',
  'global_sum',
  'none',
#  'mean_max',
  ### reduced
  'global_att',
'global_sort',#DGCNN
  'set2set',  # a seq2seq method

]
ACT_PRIMITIVES = [
  "sigmoid", "tanh", "relu",
  "softplus", "leaky_relu", "relu6", "elu"
]
SC_PRIMITIVES = [
  'none',
  'skip',
]
LA_PRIMITIVES=[
  'l_max',
  'l_concat',
  'l_lstm',
  'l_sum',
  'l_mean',
]


