# SR-DAG

example:

1. discrete datas:(Asia)

  BDeu score:

  python main.py --data asia --nodes_num 8 --size 4096 --score BDeu

  BIC score:

  python main.py --data asia --nodes_num 8 --size 4096 --score BIC


2. continue datas:

  python main.py --data DATA --nodes_num NODES_NUM --size SAMPLED_SIZE --score GaussBIC

3. store model:

  python main.py --data DATA --nodes_num NODES_NUM --size SAMPLED_SIZE --score BDeu/BIC/GaussBIC --store

4. train based on stored model:

  python main.py --data DATA --nodes_num NODES_NUM --size SAMPLED_SIZE --score BDeu/BIC/GaussBIC --valid
