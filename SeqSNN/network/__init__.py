from .base import NETWORKS

from .ann.tsrnn import TSRNN
from .ann.itransformer import ITransformer
from .ann.tcn2d import TemporalConvNet2D, TemporalBlock2D

from .snn.snn import TSSNN, TSSNN2D, ITSSNN2D
from .snn.spike_tcn import SpikeTemporalConvNet2D, SpikeTemporalBlock2D
from .snn.ispikformer import iSpikformer
from .snn.spikformer import Spikformer
from .snn.spikformer_CPG import SpikformerCPG
from .snn.spikernn import SpikeRNN, SpikeRNN2D
