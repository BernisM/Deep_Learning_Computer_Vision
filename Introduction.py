import numpy as np 
from numpy import genfromtxt

fld = r'C:\Users\massw\OneDrive\Bureau\Programmation\Python_R\Computer-Vision-with-Python\DATA'
file = r"bank_note_data.txt"
path = '{}\{}'.format(fld,file)

data = genfromtxt(path,delimiter=',')

data

labels = data[:,4]
