import imageio
import matplotlib.pyplot as plt

import numpy as np

pic = imageio.imread('parrot.jpg')

ber=[]
SNR=[-15,-10,-5,0,3,5,10]

a = pic.shape[0]
b = pic.shape[1]
c=3
plt.figure(figsize = (a,b))
plt.imshow(pic)
plt.show()
def signal_to_binary(signal):
    
    bin_array=[]
    from_x=0
    x=0
    to_x=20
    y=0
    for j in signal[0:500]:
        if abs(j)>x:
            x=abs(j)
    limit=x

      
    
    
    for i in signal:
        x=0
        for j in signal[from_x:to_x]:
            x=x+abs(j)
        if x>limit*4:
            bin_array.append('1')
        else:
            bin_array.append('0')
        from_x=from_x+20
        to_x=to_x+20
    return ''.join(bin_array)


def image_to_binary(image): # convert 3D image array to 1D binary array
    arr = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                arr.append(int_bin(image[i, j, k]))
    arr = ''.join(arr)
    return arr


def int_bin(value): # convert integer to binary 
    num = bin(value)
    new_num = num[2:]
    if len(new_num) < 8:
        new_num = '0'*(8-len(new_num)) + new_num
    return new_num

def binary_to_signal(arr): # convert binary data to analog signal
    x = np.arange(0, 2, 0.1, 'float16')
    
    y0 = np.sin(np.pi*x)
    y0 = y0/(np.sqrt(np.mean(y0**2)))
    
    y1 = np.zeros(20)
    
    
    signal = np.empty([0], 'float16')
    for i in arr:
        if i == '0':
            signal = np.concatenate([signal, y1])
        else:
            signal = np.concatenate([signal, y0])
    return signal

def add_noise(sig, SNR):
    sigma = np.sqrt(0.5*10**(-SNR/10)) #Formula of variance given by Professor
    noise = np.random.normal(0, sigma, len(sig))
    sig = sig + noise
    return sig


def binary_to_image(arr): #convert 1D binary array to 3D image array and 
    img_rec = np.arange(a*b*c).reshape(a, b, c)
    increment = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                img_rec[i, j, k] = int(arr[increment : increment + 8], 2)
                increment = increment + 8
    return img_rec

def ber_calc(original, corrupt): # Calculate bit error ratio
    count = 0
    for i in range(len(original)):
        if original[i] != corrupt[i]:
            count += 1
    ber = count/len(original)
    return ber


def main(db):
    binary_arr=image_to_binary(pic)
    signal=binary_to_signal(binary_arr)
    plt.plot(np.arange(0, 6, 0.1, 'float16'), signal[:60])
    plt.title(str(db))
    plt.show()
    dirty_signal=add_noise(signal,db)
    plt.plot(dirty_signal[:100])
    plt.title(str(db))
    plt.show()
    dirty_bits=signal_to_binary(dirty_signal)
    dirty_image=binary_to_image(dirty_bits)
    error=ber_calc(binary_arr,dirty_bits)
    ber.append(error)
    plt.imshow(dirty_image)
    plt.title(str(db))
    plt.show()
for i in SNR:
    main(i)
plt.plot(SNR, ber)
plt.yscale('log')
plt.show()
ber = []

