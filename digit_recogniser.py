from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc

digits = datasets.load_digits()
features = digits.data
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features , labels)

print(features.shape)

#print(clf.predict([features[-2]]))

img = misc.imread('')
img = misc.imresize(img ,(8,8))
img = img.astype(digits.images.dtype)
img = img.bytescale(img , high = 16 , low = 8)


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test .append(sum(eachPixel/3.0))

print(clf.predict([x_test]))