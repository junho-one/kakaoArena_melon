import subprocess

# subprocess.run(["conda activate", "tran-jh"], shell=True)
python = "/home/isoft_ck01/.conda/envs/tran-jh/bin/python3.6"
# python = "python"

subprocess.run(["{} train.py --batch_size=4096 --dataset=valid --lr=0.001 --factor_num=32 --gpu=4 --num_ng=100 --epochs=8".format(python)], shell=True)

subprocess.run(["{} predict.py --batch_size=1 --dataset=valid --factor_num=32 --gpu=4 --epochs=8".format(python)], shell=True)

#subprocess.run(["{} train.py --batch_size=4096 --dataset=test --lr=0.001 --factor_num=32 --gpu=4 --num_ng=10 --epochs=8".format(python)], shell=True)

# subprocess.run(["{} predict.py --batch_size=1 --dataset=test --factor_num=32 --gpu=4 --epochs=7".format(python)], shell=True)




# subprocess.call('''python3 train.py
#                             --batch_size=1024
#                             --lr=0.001
#                             --factor_num=16
#                             --gpu=1
#                             --dataset=valid
#                             --num_ng=4
#                             --epochs=10''', shell=True)
#
# subprocess.call('''python3 predict.py
#                             --batch_size=1
#                             --dataset=valid
#                             --factor_num=16
#                             --gpu=1
#                             --epochs=10''', shell=True)

