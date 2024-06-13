import pickle as pic
import ast

# file_= 'run20240531190533.txt' # Test Run fair computational effort analysis
# file_= 'run20240603130619.txt'
# with open(file_, 'rb') as fh:
#     information=fh.read().decode('utf-8')
#     information_list=ast.literal_eval(information)



file_2 = 'run20240601060629.txt'# Test Run fair computational effort analysis
file_2='run20240602070653.txt'# Test Run fair computational effort analysis
file_2='run20240602090618.txt'# Test Run fair computational effort analysis
file_2='run20240602140619.txt'# Test Run fair computational effort analysis
file_2='run20240604030642.txt'# test Run fair computational effort analysis (90,5,L)
file_2='run20240604080601.txt'# test Run fair computational effort analysis (90,5,L)
file_2='run20240605010655.txt'
file_2='run20240606200655.txt'
file_2='run20240607020621.txt'
file_2 = 'run20240609070632.txt'# test Run fair computational effort analysis (90,5,L)
file_2 = 'run20240608230625.txt'# test Run fair computational effort fbm
file_2 = 'run20240609210633.txt' # Run fair computational effort fbm (final N_T=9)
file_2= 'run20240610040620.txt' # incomplete run (1st half)
file_2= 'run20240610090622.txt' # incomplete run (2nd half)
file_2= 'run20240610110654.txt' # Google cloud run final Final Run fair computational effort analysis (90,5,L)
file_2= 'run20240610200648.txt'# Google cloud run final Final Run fair computational effort analysis (90,10,3)
file_2='run20240611230633.txt'# test Run fair computational effort analysis (90,5,3) for N_T=49
file_2='run20240612100609.txt'# Run fair computational effort analysis (90,5,3) for N_T=49
file_2= 'run20240612140646.txt'# test Run fair computational effort analysis fBm for N_T=49
file_2='run20240612220606.txt' # high dimensional test instance
file_2 = 'run20240612220603.txt' # test instance fbm
file_2='run20240613070643.txt' # final run fbm N_T =49
with open(file_2, 'rb') as fh:
    information2=fh.read().decode('utf-8')
    information2_list=ast.literal_eval(information2)
information_all=information2_list#[*information_list, *information2_list] #information2_list#

pickle_name= file_2.rstrip('txt')+'pic'


with open(pickle_name, 'wb') as fh:
    pic.dump(information_all, fh)

