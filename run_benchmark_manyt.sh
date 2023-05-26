for f in $1/subconfigs/*.yaml
do
  echo $f
  python notebooks/benchmarking/benchmark_treetoolml.py --cfg $f
  python notebooks/benchmarking/draw_graphs_treetoolml.py --cfg $f
done