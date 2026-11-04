"""
Pure-PyTorch drop-in replacement for the subset of DGL & DGLLife used by `MVSL-DSF/Network.py`
and `MSSF_plus/baselines_rerun/run_mvsl.py`.

Why this exists:
  Google Colab's default Python 3.13 runtime has no binary DGL wheels (`data.dgl.ai` only
  builds up to Python 3.12), causing `pip install dgl` to fall back to the obsolete 2019
  `dgl==0.4.3.post2` sdist on PyPI (`ImportError: cannot import name 'Mapping' from 'collections'`).
  `MVSL-DSF/Network.py` only uses 8 fundamental message-passing primitives:
    - `dgl.graph`, `dgl.batch`, `dgl.remove_self_loop`, `dgl.add_self_loop`
    - `g.local_var()`, `g.local_scope()`, `g.apply_edges(...)`, `g.update_all(...)`
    - `dgl.nn.pytorch.edge_softmax`
    - `dgl.broadcast_nodes`, `dgl.softmax_nodes`, `dgl.sum_nodes`
    - `dgl.function.copy_e`, `dgl.function.u_mul_e`, `dgl.function.sum`
  All 8 primitives are implemented below using exact PyTorch `scatter_reduce_` / `index_add_`
  operations on GPU/CPU, plus exact RDKit featurization matching `dgllife.utils`
  (`AttentiveFPAtomFeaturizer`, `AttentiveFPBondFeaturizer`, `smiles_to_bigraph`).
"""

import contextlib
import sys
import types
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


class _LazyNodeLookup:
    """Index `ndata` tensors by edge endpoint indices (`src` or `dst`) on demand."""

    def __init__(self, ndata: Dict[str, torch.Tensor], idx: torch.Tensor):
        self._ndata = ndata
        self._idx = idx

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._ndata[key][self._idx]


class _EdgeBatchView:
    def __init__(self, graph: 'MiniDGLGraph'):
        self.src = _LazyNodeLookup(graph.ndata, graph.src)
        self.dst = _LazyNodeLookup(graph.ndata, graph.dst)
        self.data = graph.edata


class MiniDGLGraph:
    """Exact pure-PyTorch DGLGraph equivalent for AttentiveFP / MVSL-DSF."""

    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        ndata: Optional[Dict[str, torch.Tensor]] = None,
        edata: Optional[Dict[str, torch.Tensor]] = None,
        batch_num_nodes: Optional[torch.Tensor] = None,
        batch_num_edges: Optional[torch.Tensor] = None,
        node_batch_idx: Optional[torch.Tensor] = None,
    ):
        device = src.device
        self.src = src.to(dtype=torch.long)
        self.dst = dst.to(dtype=torch.long)
        self._num_nodes = int(num_nodes)
        self.ndata: Dict[str, torch.Tensor] = {} if ndata is None else dict(ndata)
        self.edata: Dict[str, torch.Tensor] = {} if edata is None else dict(edata)

        if batch_num_nodes is None:
            self._batch_num_nodes = torch.tensor([self._num_nodes], dtype=torch.long, device=device)
        else:
            self._batch_num_nodes = batch_num_nodes.to(device=device, dtype=torch.long)

        if batch_num_edges is None:
            self._batch_num_edges = torch.tensor([len(self.src)], dtype=torch.long, device=device)
        else:
            self._batch_num_edges = batch_num_edges.to(device=device, dtype=torch.long)

        if node_batch_idx is None:
            self._node_batch_idx = torch.repeat_interleave(
                torch.arange(len(self._batch_num_nodes), dtype=torch.long, device=device),
                self._batch_num_nodes,
            )
        else:
            self._node_batch_idx = node_batch_idx.to(device=device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.src.device

    @property
    def batch_size(self) -> int:
        return int(len(self._batch_num_nodes))

    def num_nodes(self) -> int:
        return self._num_nodes

    def number_of_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return int(len(self.src))

    def number_of_edges(self) -> int:
        return int(len(self.src))

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_num_nodes

    def batch_num_edges(self) -> torch.Tensor:
        return self._batch_num_edges

    def set_batch_num_nodes(self, val: torch.Tensor) -> None:
        self._batch_num_nodes = val.to(device=self.device, dtype=torch.long)
        self._node_batch_idx = torch.repeat_interleave(
            torch.arange(len(self._batch_num_nodes), dtype=torch.long, device=self.device),
            self._batch_num_nodes,
        )

    def set_batch_num_edges(self, val: torch.Tensor) -> None:
        self._batch_num_edges = val.to(device=self.device, dtype=torch.long)

    def add_nodes(self, num: int) -> None:
        self._num_nodes += int(num)
        self._batch_num_nodes = torch.tensor([self._num_nodes], dtype=torch.long, device=self.device)
        self._node_batch_idx = torch.zeros(self._num_nodes, dtype=torch.long, device=self.device)

    def to(self, device: Union[str, torch.device]) -> 'MiniDGLGraph':
        dev = torch.device(device)
        return MiniDGLGraph(
            src=self.src.to(dev),
            dst=self.dst.to(dev),
            num_nodes=self._num_nodes,
            ndata={k: v.to(dev) for k, v in self.ndata.items()},
            edata={k: v.to(dev) for k, v in self.edata.items()},
            batch_num_nodes=self._batch_num_nodes.to(dev),
            batch_num_edges=self._batch_num_edges.to(dev),
            node_batch_idx=self._node_batch_idx.to(dev),
        )

    def local_var(self) -> 'MiniDGLGraph':
        return MiniDGLGraph(
            src=self.src,
            dst=self.dst,
            num_nodes=self._num_nodes,
            ndata=dict(self.ndata),
            edata=dict(self.edata),
            batch_num_nodes=self._batch_num_nodes,
            batch_num_edges=self._batch_num_edges,
            node_batch_idx=self._node_batch_idx,
        )

    @contextlib.contextmanager
    def local_scope(self):
        old_ndata = dict(self.ndata)
        old_edata = dict(self.edata)
        try:
            yield self
        finally:
            self.ndata = old_ndata
            self.edata = old_edata

    def apply_edges(self, func: Callable) -> None:
        view = _EdgeBatchView(self)
        new_edata = func(view)
        if isinstance(new_edata, dict):
            self.edata.update(new_edata)

    def update_all(self, message_func: Callable, reduce_func: Callable) -> None:
        messages = message_func(self)
        reduced = reduce_func(self, messages)
        if isinstance(reduced, dict):
            self.ndata.update(reduced)

    def remove_self_loop(self) -> 'MiniDGLGraph':
        return remove_self_loop(self)

    def add_self_loop(self) -> 'MiniDGLGraph':
        return add_self_loop(self)

    def clone(self) -> 'MiniDGLGraph':
        return MiniDGLGraph(
            src=self.src,
            dst=self.dst,
            num_nodes=self._num_nodes,
            ndata=dict(self.ndata),
            edata=dict(self.edata),
            batch_num_nodes=self._batch_num_nodes.clone(),
            batch_num_edges=self._batch_num_edges.clone(),
        )


# ============================================================
# DGL Graph Construction & Batching Functions
# ============================================================

def graph(
    data: Tuple[Iterable[int], Iterable[int]],
    num_nodes: Optional[int] = None,
    idtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> MiniDGLGraph:
    u, v = data
    dev = torch.device(device) if device is not None else torch.device('cpu')
    src = torch.as_tensor(list(u) if not isinstance(u, (torch.Tensor, np.ndarray)) else u, dtype=torch.long, device=dev).flatten()
    dst = torch.as_tensor(list(v) if not isinstance(v, (torch.Tensor, np.ndarray)) else v, dtype=torch.long, device=dev).flatten()
    if num_nodes is None:
        if len(src) > 0:
            num_nodes = int(max(int(src.max().item()), int(dst.max().item())) + 1)
        else:
            num_nodes = 0
    return MiniDGLGraph(src=src, dst=dst, num_nodes=int(num_nodes))


def remove_self_loop(g: MiniDGLGraph) -> MiniDGLGraph:
    mask = g.src != g.dst
    new_edata = {k: v[mask] for k, v in g.edata.items()}
    return MiniDGLGraph(
        src=g.src[mask],
        dst=g.dst[mask],
        num_nodes=g._num_nodes,
        ndata=dict(g.ndata),
        edata=new_edata,
    )


def add_self_loop(g: MiniDGLGraph) -> MiniDGLGraph:
    loop_nodes = torch.arange(g._num_nodes, dtype=torch.long, device=g.device)
    new_src = torch.cat([g.src, loop_nodes], dim=0)
    new_dst = torch.cat([g.dst, loop_nodes], dim=0)
    new_edata = {}
    for k, v in g.edata.items():
        pad = torch.zeros((g._num_nodes, *v.shape[1:]), dtype=v.dtype, device=v.device)
        new_edata[k] = torch.cat([v, pad], dim=0)
    return MiniDGLGraph(
        src=new_src,
        dst=new_dst,
        num_nodes=g._num_nodes,
        ndata=dict(g.ndata),
        edata=new_edata,
    )


def batch(graphs: List[MiniDGLGraph]) -> MiniDGLGraph:
    if len(graphs) == 0:
        raise ValueError("Cannot batch an empty list of graphs.")
    dev = graphs[0].device
    num_nodes_list = [g._num_nodes for g in graphs]
    num_edges_list = [len(g.src) for g in graphs]

    offsets = np.cumsum([0] + num_nodes_list[:-1])
    src_list = [g.src + int(off) for g, off in zip(graphs, offsets)]
    dst_list = [g.dst + int(off) for g, off in zip(graphs, offsets)]

    batched_src = torch.cat(src_list, dim=0) if src_list else torch.empty(0, dtype=torch.long, device=dev)
    batched_dst = torch.cat(dst_list, dim=0) if dst_list else torch.empty(0, dtype=torch.long, device=dev)

    batched_ndata = {}
    for k in graphs[0].ndata.keys():
        batched_ndata[k] = torch.cat([g.ndata[k] for g in graphs], dim=0)

    batched_edata = {}
    for k in graphs[0].edata.keys():
        batched_edata[k] = torch.cat([g.edata[k] for g in graphs], dim=0)

    b_nnodes = torch.tensor(num_nodes_list, dtype=torch.long, device=dev)
    b_nedges = torch.tensor(num_edges_list, dtype=torch.long, device=dev)
    node_batch_idx = torch.repeat_interleave(
        torch.arange(len(graphs), dtype=torch.long, device=dev),
        b_nnodes,
    )

    return MiniDGLGraph(
        src=batched_src,
        dst=batched_dst,
        num_nodes=int(sum(num_nodes_list)),
        ndata=batched_ndata,
        edata=batched_edata,
        batch_num_nodes=b_nnodes,
        batch_num_edges=b_nedges,
        node_batch_idx=node_batch_idx,
    )


# ============================================================
# DGL Readout & Edge Softmax Primitives
# ============================================================

def broadcast_nodes(g: MiniDGLGraph, graph_feat: torch.Tensor) -> torch.Tensor:
    return graph_feat[g._node_batch_idx]


def softmax_nodes(g: MiniDGLGraph, feat_name: str) -> torch.Tensor:
    x = g.ndata[feat_name]
    b_size = g.batch_size
    b_idx = g._node_batch_idx.view(-1, *([1] * (x.ndim - 1))).expand_as(x)
    max_per_g = torch.full((b_size, *x.shape[1:]), -1e38, dtype=x.dtype, device=x.device)
    max_per_g.scatter_reduce_(0, b_idx, x, reduce='amax', include_self=False)
    exp_x = torch.exp(x - max_per_g[g._node_batch_idx])
    denom = torch.zeros((b_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    denom.scatter_add_(0, b_idx, exp_x)
    return exp_x / (denom[g._node_batch_idx] + 1e-12)


def sum_nodes(g: MiniDGLGraph, feat_name: str, weight_name: Optional[str] = None) -> torch.Tensor:
    x = g.ndata[feat_name]
    if weight_name is not None:
        x = x * g.ndata[weight_name]
    out = torch.zeros((g.batch_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    out.index_add_(0, g._node_batch_idx, x)
    return out


def edge_softmax(g: MiniDGLGraph, logits: torch.Tensor) -> torch.Tensor:
    x = logits
    n_nodes = g._num_nodes
    dst_idx = g.dst.view(-1, *([1] * (x.ndim - 1))).expand_as(x)
    max_per_dst = torch.full((n_nodes, *x.shape[1:]), -1e38, dtype=x.dtype, device=x.device)
    max_per_dst.scatter_reduce_(0, dst_idx, x, reduce='amax', include_self=False)
    exp_x = torch.exp(x - max_per_dst[g.dst])
    denom = torch.zeros((n_nodes, *x.shape[1:]), dtype=x.dtype, device=x.device)
    denom.scatter_add_(0, dst_idx, exp_x)
    return exp_x / (denom[g.dst] + 1e-12)


# ============================================================
# dgl.function Primitives
# ============================================================

def copy_e(e_field: str, m_field: str) -> Callable:
    def _msg(g: MiniDGLGraph) -> Dict[str, torch.Tensor]:
        return {m_field: g.edata[e_field]}
    return _msg


def u_mul_e(u_field: str, e_field: str, m_field: str) -> Callable:
    def _msg(g: MiniDGLGraph) -> Dict[str, torch.Tensor]:
        return {m_field: g.ndata[u_field][g.src] * g.edata[e_field]}
    return _msg


def fn_sum(m_field: str, out_field: str) -> Callable:
    def _red(g: MiniDGLGraph, messages: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        m = messages[m_field]
        out = torch.zeros((g._num_nodes, *m.shape[1:]), dtype=m.dtype, device=m.device)
        out.index_add_(0, g.dst, m)
        return {out_field: out}
    return _red


# ============================================================
# Exact AttentiveFP Featurizers (Xiong et al. 2019 / dgllife)
# ============================================================

_ATOM_SYMBOLS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']


def _one_hot(val, allowable_set: list, encode_unknown: bool = False) -> List[int]:
    if encode_unknown:
        if val not in allowable_set:
            return [0] * len(allowable_set) + [1]
        return [int(val == s) for s in allowable_set] + [0]
    return [int(val == s) for s in allowable_set]


class AttentiveFPAtomFeaturizer:
    """
    Exact 39-dimensional atom featurizer from `dgllife.utils.AttentiveFPAtomFeaturizer`:
      - Atom symbol one-hot (15 + 1 unknown = 16)
      - Degree one-hot [0, 1, 2, 3, 4, 5] (6)
      - Formal charge (1)
      - Num radical electrons (1)
      - Hybridization one-hot [SP, SP2, SP3, SP3D, SP3D2] + unknown (6)
      - Is aromatic (1)
      - Total num Hs one-hot [0, 1, 2, 3, 4] (5)
      - Is chiral center (1)
      - Chirality type one-hot [R, S] (2)
    Total = 16 + 6 + 1 + 1 + 6 + 1 + 5 + 1 + 2 = 39.
    """

    def __init__(self, atom_data_field: str = 'h'):
        self.atom_data_field = atom_data_field

    def feat_size(self) -> int:
        return 39

    def __call__(self, mol) -> Dict[str, torch.Tensor]:
        from rdkit import Chem
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            pass

        hyb_set = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
        feats = []
        for atom in mol.GetAtoms():
            f = []
            f.extend(_one_hot(atom.GetSymbol(), _ATOM_SYMBOLS, encode_unknown=True))  # 16
            f.extend(_one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5], encode_unknown=False))  # 6
            f.append(float(atom.GetFormalCharge()))  # 1
            f.append(float(atom.GetNumRadicalElectrons()))  # 1
            f.extend(_one_hot(atom.GetHybridization(), hyb_set, encode_unknown=True))  # 6
            f.append(float(atom.GetIsAromatic()))  # 1
            f.extend(_one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], encode_unknown=False))  # 5
            has_chiral = atom.HasProp('_ChiralityPossible')
            f.append(float(has_chiral))  # 1
            chiral_tag = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
            f.extend(_one_hot(chiral_tag, ['R', 'S'], encode_unknown=False))  # 2
            feats.append(f)

        arr = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 39), dtype=np.float32)
        return {self.atom_data_field: torch.from_numpy(arr)}


class AttentiveFPBondFeaturizer:
    """
    Exact 11-dimensional (with self_loop=True) bond featurizer from `dgllife.utils.AttentiveFPBondFeaturizer`:
      - Bond type one-hot [SINGLE, DOUBLE, TRIPLE, AROMATIC] (4)
      - Is conjugated (1)
      - Is in ring (1)
      - Stereo one-hot [STEREONONE, STEREOANY, STEREOZ, STEREOE] (4)
      - Is self loop (1 when self_loop=True)
    Total = 4 + 1 + 1 + 4 + 1 = 11.
    """

    def __init__(self, bond_data_field: str = 'e', self_loop: bool = False):
        self.bond_data_field = bond_data_field
        self.self_loop = self_loop

    def feat_size(self) -> int:
        return 11 if self.self_loop else 10

    def __call__(self, mol) -> Dict[str, torch.Tensor]:
        from rdkit import Chem
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        stereo_types = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
        feats = []
        for bond in mol.GetBonds():
            f = []
            f.extend(_one_hot(bond.GetBondType(), bond_types, encode_unknown=False))  # 4
            f.append(float(bond.GetIsConjugated()))  # 1
            f.append(float(bond.IsInRing()))  # 1
            f.extend(_one_hot(bond.GetStereo(), stereo_types, encode_unknown=False))  # 4
            if self.self_loop:
                f.append(0.0)  # 1
            # Directed bigraph adds two directed edges per undirected bond
            feats.append(f)
            feats.append(list(f))

        if self.self_loop:
            num_atoms = mol.GetNumAtoms()
            for _ in range(num_atoms):
                feats.append([0.0] * 10 + [1.0])

        dim = 11 if self.self_loop else 10
        arr = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, dim), dtype=np.float32)
        return {self.bond_data_field: torch.from_numpy(arr)}


def smiles_to_bigraph(
    smiles: str,
    add_self_loop: bool = False,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
) -> Optional[MiniDGLGraph]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    g = graph((src_list, dst_list), num_nodes=num_atoms)
    if add_self_loop:
        g = g.remove_self_loop()
        g = g.add_self_loop()

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))
    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))
    return g


def install_dgl_shim(force: bool = False) -> bool:
    """
    If native `dgl` or `dgllife` is unavailable or broken (e.g. Python 3.13 on Google Colab),
    register pure-PyTorch `dgl` and `dgllife.utils` modules into `sys.modules`.
    Returns True if the shim was installed, False if native `dgl` + `dgllife` worked.
    """
    if not force:
        try:
            import dgl  # noqa: F401
            from dgl.nn.pytorch import edge_softmax as _es  # noqa: F401
            from dgllife.utils import smiles_to_bigraph as _s2b  # noqa: F401
            return False
        except Exception:
            pass

    # Clean up any partially-imported broken dgl modules from sys.modules
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'dgl' or mod_name.startswith('dgl.') or mod_name == 'dgllife' or mod_name.startswith('dgllife.'):
            del sys.modules[mod_name]

    dgl_mod = types.ModuleType('dgl')
    dgl_mod.__version__ = '2.4.0-pytorch-shim'
    dgl_mod.DGLGraph = MiniDGLGraph
    dgl_mod.graph = graph
    dgl_mod.batch = batch
    dgl_mod.remove_self_loop = remove_self_loop
    dgl_mod.add_self_loop = add_self_loop
    dgl_mod.broadcast_nodes = broadcast_nodes
    dgl_mod.softmax_nodes = softmax_nodes
    dgl_mod.sum_nodes = sum_nodes

    fn_mod = types.ModuleType('dgl.function')
    fn_mod.copy_e = copy_e
    fn_mod.copy_edge = copy_e
    fn_mod.u_mul_e = u_mul_e
    fn_mod.src_mul_edge = u_mul_e
    fn_mod.sum = fn_sum
    dgl_mod.function = fn_mod

    nn_mod = types.ModuleType('dgl.nn')
    nn_pt_mod = types.ModuleType('dgl.nn.pytorch')
    nn_pt_mod.edge_softmax = edge_softmax
    nn_mod.pytorch = nn_pt_mod
    dgl_mod.nn = nn_mod

    dgllife_mod = types.ModuleType('dgllife')
    dgllife_utils_mod = types.ModuleType('dgllife.utils')
    dgllife_utils_mod.smiles_to_bigraph = smiles_to_bigraph
    dgllife_utils_mod.AttentiveFPAtomFeaturizer = AttentiveFPAtomFeaturizer
    dgllife_utils_mod.AttentiveFPBondFeaturizer = AttentiveFPBondFeaturizer
    dgllife_mod.utils = dgllife_utils_mod

    sys.modules['dgl'] = dgl_mod
    sys.modules['dgl.function'] = fn_mod
    sys.modules['dgl.nn'] = nn_mod
    sys.modules['dgl.nn.pytorch'] = nn_pt_mod
    sys.modules['dgllife'] = dgllife_mod
    sys.modules['dgllife.utils'] = dgllife_utils_mod
    return True
