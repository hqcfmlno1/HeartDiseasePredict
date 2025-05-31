import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Hard Margin SVM: Vary gamma from 0.1 to 5 with step 0.1
gammas = np.arange(0.1, 5.1, 0.1)
hard_margin_accuracies = {'sigmoid': [], 'poly': [], 'rbf': []}

print("Computing Hard Margin SVM Accuracies...")
for kernel in ('sigmoid', 'poly', 'rbf'):
    for gamma in gammas:
        clf = svm.SVC(kernel=kernel, gamma=gamma, coef0=0)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        hard_margin_accuracies[kernel].append(acc)
        print(f"Hard Margin, Kernel: {kernel}, Gamma: {gamma:.1f}, Accuracy: {acc:.4f}")

# Soft Margin SVM: Vary C from 1 to 50 with step 1
Cs = np.arange(1, 51, 1)
soft_margin_accuracies = {'sigmoid': [], 'poly': [], 'rbf': []}

print("\nComputing Soft Margin SVM Accuracies...")
for kernel in ('sigmoid', 'poly', 'rbf'):
    for C in Cs:
        clf = svm.SVC(kernel=kernel, gamma=0.5, coef0=0, C=C)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        soft_margin_accuracies[kernel].append(acc)
        print(f"Soft Margin, Kernel: {kernel}, C: {C}, Accuracy: {acc:.4f}")

# Plot Hard Margin SVM Accuracy vs. Gamma
plt.figure(figsize=(10, 6))  # Create a new figure for Hard Margin
plt.plot(gammas, hard_margin_accuracies['sigmoid'], label='Sigmoid', color='#FF6384', linewidth=2)
plt.plot(gammas, hard_margin_accuracies['poly'], label='Polynomial', color='#36A2EB', linewidth=2)
plt.plot(gammas, hard_margin_accuracies['rbf'], label='RBF', color='#4BC0C0', linewidth=2)
plt.xlabel('Gamma', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Hard Margin SVM Accuracy vs. Gamma', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0, 1)
plt.show()  # Display the first plot

# Plot Soft Margin SVM Accuracy vs. C
plt.figure(figsize=(10, 6))  # Create a new figure for Soft Margin
plt.plot(Cs, soft_margin_accuracies['sigmoid'], label='Sigmoid', color='#FF6384', linewidth=2)
plt.plot(Cs, soft_margin_accuracies['poly'], label='Polynomial', color='#36A2EB', linewidth=2)
plt.plot(Cs, soft_margin_accuracies['rbf'], label='RBF', color='#4BC0C0', linewidth=2)
plt.xlabel('C', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Soft Margin SVM Accuracy vs. C', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0, 1)
plt.show()  # Display the second plot
