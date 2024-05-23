import os
import sys
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils
from tabulate import tabulate
import pickle as pic

file_ = 'run20240418180419.pic' #Ms
file_ = 'run20240502160503.pic'
#file_ ='run20240416210454.pic' #fBM.
file_ = 'run20240503200517.pic' # Independent set (pca3)
file_= 'run20240503220535.pic' # 200 no pca
file_= 'run20240503230502.pic' # pca 1
file_='run20240511130508.pic'

# file_='run20240511150517.pic'
# file_='run20240511180558.pic'
with open(file_, 'rb') as fh:
    information=pic.load(fh)
FBM = False#max([i[2] for i in information])<=1 and min([i[2] for i in information])>=0
if FBM:
    table_ = tabulate(utils.information_format_fbm(information), headers=utils.header_fbm, tablefmt="latex_raw", floatfmt=".4f")
else:
    table_ = tabulate(utils.information_format(information), headers=utils.header_, tablefmt="latex_raw", floatfmt=".4f")
print(table_)

# information =  [[i[0]+' Cor.', *i[1:]] for i in information]

# file_2 ='run20240416100435.pic' #fBM.
# with open(file_2, 'rb') as fh:
#     information2=pic.load(fh)
# information_merged = [*information2, *information]


# with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
#     pic.dump(information_merged, fh)
