import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='co') #visible (co), ir or visible+ir (coir)
parser.add_argument("--input_file", default='./data/annotations/annotation512.txt') #path to original annotation file
parser.add_argument("--split", default='1000') #number of samples in training set others will be in validation set
flags = parser.parse_args()
prefix='./data/' #root folder of images path
if flags.mode=='coir':
    prefix='../data/'
    flags.mode = 'ir'
split = int(flags.split)

#reshape box
def getBoxCoordinates(array):
    Xs = [float(e) for e in array[4:8]]
    Ys = [float(e) for e in array[8:12]]
    tmp_Xmin = min(max(min(Xs), 0), 512)
    tmp_Xmax = min(max(max(Xs), 0), 512)
    tmp_Ymin = min(max(min(Ys), 0), 512)
    tmp_Ymax = min(max(max(Ys), 0), 512)
    return str(tmp_Xmin), str(tmp_Xmax), str(tmp_Ymin), str(tmp_Ymax)

#regroup few represented classes
#select top 5 classes
def map_class(class_in):

    if class_in == "1": return str(2) #SLV?
    if class_in == "2": return str(4) #Camping cars?
    if class_in == "4": return str(4) #Others?
    if class_in == "5": return str(3) #LLV?
    if class_in == "7": return str(4) #Vans?
    if class_in == "8": return str(4) #Planes?
    if class_in == "9": return str(4) #Tractors?
    if class_in == "10": return str(0) #Cars?
    if class_in == "23": return str(4) #Trucks?
    if class_in == "31": return str(4) #Boats?
    if class_in == "11": return str(1) #Pickups?
    print("unexisting class: " + class_in)
    return class_in

#reshape annotations
#write in file /anntations/train.txt
with open(flags.input_file, 'r') as fr:
    with open('./data/annotations/train.txt', 'w') as fw:
        count = 0
        ptr = '00000000'
        cnt = 0
        chars = [prefix + flags.mode + '512/' + ptr + '_' + flags.mode + '.png']
        line = fr.readline()
        while line and count <= split:
            new_chars = line.strip().split(" ")
            if len(new_chars) != 15: print("wrong number of element in line :", cnt)
            if new_chars[0] == ptr:
                Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                new_chars = [map_class(new_chars[12]), Xmin, Ymin, Xmax, Ymax]
                chars.extend(new_chars)
            else:
                new_line = ' '.join(chars) + "\n"
                fw.write(new_line)
                ptr = new_chars[0]
                idx = prefix + flags.mode + '512/' + ptr + '_' + flags.mode + '.png'
                Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                chars = [idx, map_class(new_chars[12]), Xmin, Ymin, Xmax, Ymax]
                count += 1
            cnt += 1
            print(cnt)
            line = fr.readline()
    #write in file /anntations/val.txt
    with open('./data/annotations/val.txt', 'w') as fw:
        while line:
            new_chars = line.strip().split(" ")
            if len(new_chars) != 15: print("wrong number of element in line :", cnt)
            if new_chars[0] == ptr:
                Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                new_chars = [map_class(new_chars[12]), Xmin, Ymin, Xmax, Ymax]
                chars.extend(new_chars)
            else:
                new_line = ' '.join(chars) + "\n"
                fw.write(new_line)
                ptr = new_chars[0]
                idx = prefix + flags.mode + '512/' + ptr + '_' + flags.mode + '.png'
                Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                chars = [idx, map_class(new_chars[12]), Xmin, Ymin, Xmax, Ymax]
            cnt += 1
            print(cnt)
            line = fr.readline()
        #don't forget last line
        new_line = ' '.join(chars) + "\n"
        fw.write(new_line)

print("total number of lines read:", cnt - 1)
