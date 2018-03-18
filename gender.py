
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score

classifiers = [tree.DecisionTreeClassifier(),KNeighborsClassifier(),GaussianProcessClassifier()]

classifierNames = ["Decision Tree","Nearest Neighbors","Gaussian Process"]

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


bestScore = 0
bestCLF = None
bestCLFname = None

for index,classifier in enumerate(classifiers):
  clf = classifier
  clf = clf.fit(X, Y)
  XPrediction = clf.predict(X)
  accuracy = accuracy_score(XPrediction,Y)
  print("Model accuracy: "+str(accuracy)+" "+str(classifierNames[index]))
  if accuracy > bestScore:
    bestScore = accuracy
    bestCLFname = classifierNames[index]
    bestCLF = clf    
  
prediction = clf.predict([[190, 70, 43]])

print("Best model: "+bestCLFname)
print(prediction)

