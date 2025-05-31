import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/heart.csv')

X = data.drop('HeartDisease', axis=1)  
y = data['HeartDisease']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  
    stratify=y,     
    random_state=42 
)

# vì các đặc trưng có thang đo khác nhau ví dụ như tuổi (0-100) còn các thuộc tính khác thường có giá trị bé từ 0 1 2 nên khi 
# giữ nguyên các giá trị như vậy mà không scale lại thì sẽ gây ra sai lệch 
# ví dụ như trong kernal rbf có đề cập đến khoảng cách vector nếu không scale lại mà để tuổi lớn như vậy thì sẽ cho kết quả có thể trái với kì vọng

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hard margin SVM:
print("Accuracy of Kernal Hard Margin SVM:")
for kernel in ('sigmoid', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=0.5, coef0 = 0)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy of "+kernel+" kernal fuction:", accuracy_score(y_test, y_pred))

# Soft margin SVM:
print("Accuracy of Kernal Soft Margin SVM:")
for kernel in ('sigmoid', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=0.5, coef0 = 0, C=0.1)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy of "+kernel+" kernal fuction:", accuracy_score(y_test, y_pred))

