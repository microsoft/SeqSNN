import SeqSNN.module.encoding.snntorch as snn
import SeqSNN.module.encoding.spikingjelly as sj

SpikeEncoder = {
    "snntorch": {
        "repeat": snn.encoder.RepeatEncoder,
        "conv": snn.encoder.ConvEncoder,
        "delta": snn.encoder.DeltaEncoder,
    },
    "spikingjelly": {
        "repeat": sj.encoder.RepeatEncoder,
        "conv": sj.encoder.ConvEncoder,
        "delta": sj.encoder.DeltaEncoder,
    },
}
