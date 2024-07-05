RUN

python -u ressl_gram_third.py --dataset cross1      --k 4096  --m 0.99


TEST

python -u linear_eval.py --dataset cross1      --checkpoint bgt-cross1.pth       --s cos
