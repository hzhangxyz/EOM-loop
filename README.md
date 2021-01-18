# EOM loop

EOM loop simulator

## Usage

Here is some examples

```
python main.py 后选择成功率 -l=50 -n=8 --r=0.8 --omega=0.8 --phi=0 --psi=0 --post=4 --post_last=0,2
```

```
python main.py 截断收敛性 -l=50 --r=0.8 --omega=0.8 --phi=0 --psi=0 -n=8 -c=4
```

```
python main.py 误差大小 -l=50 -n=8 --r=0.8 --omega=0.8 --phi=0 --psi=0 --eta=0.99 --delta_omega=0.01 --post=4
```
