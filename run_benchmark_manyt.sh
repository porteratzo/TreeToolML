for f in $1/subconfigs/*.yaml
do
  echo $f
  python notebooks/benchmarking/benchmark_filter.py --cfg $f
  python notebooks/benchmarking/draw_graphs_filter.py --cfg $f
done