import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from sklearn.decomposition import FastICA, PCA

#repr of wav files as global variables
one = read('mic1.wav')
two = read('mic2.wav')
three = read('mic3.wav')
four = read('mic4.wav')
five = read('mic5.wav')


class MyICA(object):
    def __init__(self, fun, centering, whiten):
        self.fun = fun
        self.centering = centering
        self.whiten = whiten
        self.res = np.inf

        #store wav files as class var
        self.rate, self.s1 = one
        self.rate, self.s2 = two
        self.rate, self.s3 = three
        self.rate, self.s4 = four
        self.rate, self.s5 = five

        #generate observation matrix
        if self.centering:
            self.s1 = self.s1 - np.mean(self.s1, dtype=np.float64)
            self.s2 = self.s2 - np.mean(self.s2, dtype=np.float64)
            self.s3 = self.s3 - np.mean(self.s3, dtype=np.float64)
            self.s4 = self.s4 - np.mean(self.s4, dtype=np.float64)
            self.s5 = self.s5 - np.mean(self.s5, dtype=np.float64)
        self.X = np.c_[self.s1, self.s2, self.s3, self.s4, self.s5]

    def run(self):
        #compute ica for 5 components
        ica = FastICA(n_components=5, algorithm='parallel', whiten=self.whiten, fun=self.fun)
        #reconstruct signals
        tempS_ = ica.fit_transform(self.X)
        #get reconstructed mixing matrix
        tempA_ = ica.mixing_
        temprecon = np.dot(tempS_, tempA_)
        #get residual from observation matrix
        tempres = np.linalg.norm(self.X - temprecon)
        #if residual is smaller than the other runs, keep them
        if tempres < self.res:
            self.S_ = tempS_
            self.A_ = tempA_
            self.recon = temprecon
            self.res = tempres
        return tempres

    def save(self):
	#print residual matrix
	temprecon = np.dot(self.S_, self.A_)
	tempres = self.X - temprecon
	print 'residual:'
	print tempres.T

        #save independent components to a wav file
        write('shat1.wav', self.rate, self.S_[:, 0])
        write('shat2.wav', self.rate, self.S_[:, 1])
        write('shat3.wav', self.rate, self.S_[:, 2])
        write('shat4.wav', self.rate, self.S_[:, 3])
        write('shat5.wav', self.rate, self.S_[:, 4])

        #save reconstructed mixture signals
        write('recon1.wav', self.rate, self.recon[:, 0])
        write('recon2.wav', self.rate, self.recon[:, 1])
        write('recon3.wav', self.rate, self.recon[:, 2])
        write('recon4.wav', self.rate, self.recon[:, 3])
        write('recon5.wav', self.rate, self.recon[:, 4])

    def __repr__(self):
        strf = str(self.fun)
        if callable(strf):
            strf = 'g3'
        return 'ICA: fun - ' + strf + ', centering - ' + str(self.centering) + ', whiten - ' + str(self.whiten) + ' : ' + str(self.res)

def g3(x):
    return 0.25 * x ** 4, (x ** 3).mean(axis=-1)

if __name__ == '__main__':
    runs = 5
    #MyICA(fun, centering, whiten)
    icas = [MyICA('logcosh', True, True),
        MyICA('logcosh', True, False),
        MyICA('logcosh', False, True),
        MyICA('logcosh', False, False),
        MyICA('exp', True, True),
        MyICA('exp', True, False),
        MyICA('exp', False, True),
        MyICA('exp', False, False),
        MyICA(g3, True, True),
        MyICA(g3, True, False),
        MyICA(g3, False, True),
        MyICA(g3, False, False)]
    residuals = np.empty((len(icas), runs))

    for i in range(len(icas)):
        print i
        for j in range(runs):
            print j,
            residuals[i][j] = icas[i].run()
        print

    means = np.mean(residuals, axis=1)

    #get the ica with the lowest residual and save the wav files
    ica = icas[np.argmin(means)]
    ica.save()
    print ica
